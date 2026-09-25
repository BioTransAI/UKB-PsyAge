# =============================================================================
# Psychological age in UK Biobank
# Analysis workflow accompanying the manuscript
# =============================================================================
# The workflow contains age prediction, feature interpretation, phenotype,
# and disease and mortality analyses.
# Statistical calculations and parameter settings are retained from the supplied
# analysis. Figure construction, plot formatting, and image exports are omitted.
# SHAP contributions, spline fits, and numerical prediction tables are retained.
#
# Replace descriptive <PLACEHOLDERS> with the corresponding local resources.
# Repeated placeholders refer to the same input or output. Public UK Biobank
# annotation URLs are retained for the schema and ICD mapping steps.
# The feature-importance and SHAP block uses the LightGBM model interface.
# See read.md for the module overview and input requirements.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Input checks and candidate regression learners
# -----------------------------------------------------------------------------
datall=readRDS('<ANALYSIS_DATA_RDS>')
data=datall$q_final$mhq2016
suppressPackageStartupMessages({
  library(data.table); library(mlr3); library(mlr3learners); library(mlr3extralearners)
})
SEED <- 20260815L
C <- as.integer(Sys.getenv("ML_CORES","8"))
dir.create("<MODEL_OUTPUT_DIRECTORY>", FALSE)

# ==================== QC + data ====================
stopifnot("label" %in% names(data), !anyNA(data), !anyDuplicated(rownames(data)),
          is.numeric(data$label), all(vapply(data[-1], is.numeric, logical(1))))

# Remove zero-variance features.
nz <- vapply(data[-1], \(x) length(unique(x)) > 1L, logical(1))
removed_constant <- names(data)[-1][!nz]
d <- data[,c(TRUE,nz),drop=FALSE]

# Map UK Biobank feature names to x1, x2, ... and retain the mapping.
feature_map <- data.table(original=names(d)[-1], feature=paste0("x",seq_len(ncol(d)-1)))
names(d)[-1] <- feature_map$feature
eid <- rownames(d)

task <- as_task_regr(as.data.table(d), target="label", id="age")

# ==================== Check learner availability ====================
need <- c("regr.lm","regr.cv_glmnet","regr.earth","regr.cubist","regr.rpart",
          "regr.ctree","regr.liblinear","regr.ranger","regr.gbm",
          "regr.xgboost","regr.lightgbm")
miss <- setdiff(need, mlr_learners$keys())
if(length(miss)) stop("Unavailable learners: ",paste(miss,collapse=", "))

# ==================== 20 candidates ====================
L <- list(
  lrn("regr.lm",id="LM"),
  lrn("regr.cv_glmnet",id="Ridge",alpha=0,nfolds=5,type.measure="mae"),
  lrn("regr.cv_glmnet",id="LASSO",alpha=1,nfolds=5,type.measure="mae"),
  lrn("regr.cv_glmnet",id="ElasticNet",alpha=.5,nfolds=5,type.measure="mae"),
  lrn("regr.earth",id="MARS",degree=2,nprune=40),
  lrn("regr.cubist",id="Cubist",committees=50,neighbors=5),
  lrn("regr.rpart",id="CART",cp=1e-4,maxdepth=15,minsplit=100),
  lrn("regr.ctree",id="CTree",maxdepth=10,minsplit=100),
  lrn("regr.liblinear",id="SVR_L2_Primal",type=11,cost=1),
  lrn("regr.liblinear",id="SVR_L2_Dual",type=12,cost=1),
  lrn("regr.liblinear",id="SVR_L1_Dual",type=13,cost=1),
  lrn("regr.ranger",id="RF",num.trees=800,mtry.ratio=.5,min.node.size=5,
      splitrule="variance",num.threads=C),
  lrn("regr.ranger",id="ExtraTrees",num.trees=800,mtry.ratio=.7,min.node.size=5,
      splitrule="extratrees",num.random.splits=5,num.threads=C),
  lrn("regr.gbm",id="GBM",n.trees=1200,interaction.depth=3,
      shrinkage=.03,bag.fraction=.8,n.cores=C),
  lrn("regr.xgboost",id="XGB_GBT",booster="gbtree",nrounds=1000,eta=.03,
      max_depth=6,min_child_weight=10,subsample=.8,colsample_bytree=.8,
      tree_method="hist",nthread=C),
  lrn("regr.xgboost",id="XGB_DART",booster="dart",nrounds=800,eta=.03,
      max_depth=6,subsample=.8,colsample_bytree=.8,rate_drop=.05,
      tree_method="hist",nthread=C),
  lrn("regr.xgboost",id="XGB_Linear",booster="gblinear",nrounds=500,nthread=C),
  lrn("regr.lightgbm",id="LGB_GBDT",boosting="gbdt",num_iterations=1000,
      learning_rate=.03,num_leaves=31,min_data_in_leaf=50,
      feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,num_threads=C),
  lrn("regr.lightgbm",id="LGB_DART",boosting="dart",num_iterations=800,
      learning_rate=.03,num_leaves=31,min_data_in_leaf=50,
      feature_fraction=.8,num_threads=C),
  lrn("regr.lightgbm",id="LGB_RF",boosting="rf",num_iterations=800,
      num_leaves=31,min_data_in_leaf=50,feature_fraction=.8,
      bagging_fraction=.8,bagging_freq=1,num_threads=C)
)
stopifnot(length(L)==20)
ids <- vapply(L,\(x)x$id,"")

# ==================== metrics ====================
score <- function(p){
  z <- as.data.table(p); y <- z$truth; yh <- z$response
  data.table(N=length(y),MAE=mean(abs(y-yh)),RMSE=sqrt(mean((y-yh)^2)),
             R2=1-sum((y-yh)^2)/sum((y-mean(y))^2),
             PCC=cor(y,yh),Bias=mean(yh-y))
}
pred_dt <- function(p,set){
  z <- as.data.table(p)
  data.table(eid=eid[z$row_ids],set=set,truth=z$truth,
             pred=z$response,age_gap=z$response-z$truth)
}

# -----------------------------------------------------------------------------
# 2. Training cross-validation and held-out evaluation
# -----------------------------------------------------------------------------
# ============================================================
# 80/20 split
# ============================================================
set.seed(SEED)
sp <- partition(task,ratio=.8)
tr <- task$clone(deep=TRUE)
tr$filter(sp$train)
# ============================================================
# Training ONLY: same 5-fold CV for all 20 models
# best = minimum MAE
# ============================================================
set.seed(SEED)
cv5 <- rsmp("cv",folds=5)
cv5$instantiate(tr)
perf <- vector("list",length(L))
best_mae <- Inf
best_id <- NULL
best_train_pred <- NULL
for(i in seq_along(L)){
  cat(sprintf("[%02d/20] %s\n",i,L[[i]]$id))

  rr <- resample(tr,L[[i]]$clone(deep=TRUE),cv5,store_models=FALSE)
  p <- rr$prediction()
  s <- score(p)

  perf[[i]] <- cbind(model=L[[i]]$id,s)

  if(s$MAE < best_mae){
    best_mae <- s$MAE
    best_id <- L[[i]]$id
    best_train_pred <- pred_dt(p,"train_5fold_OOF")
  }
  rm(rr,p); gc()
}
train_perf <- rbindlist(perf)
setorder(train_perf,MAE,RMSE)
cat("\nBEST =",best_id,"\n")
print(train_perf)
# ============================================================
# TEST: evaluation only
# The selected model has already been fixed using training data.
# ============================================================
# ================= TEST: evaluate only the learner selected by training cross-validation =================
i_best <- match(best_id, ids)

best_model <- L[[i_best]]$clone(deep=TRUE)
best_model$train(task, row_ids=sp$train)

best_test <- best_model$predict(task, row_ids=sp$test)

test_perf <- cbind(
  model = best_id,
  score(best_test)
)

best_test_pred <- pred_dt(best_test, "test")
test_perf

# -----------------------------------------------------------------------------
# 3. Full-cohort out-of-fold predictions and final model fitting
# -----------------------------------------------------------------------------
# ============================================================
# Selected model → ALL subjects 10-fold OOF
# ============================================================
set.seed(SEED)
cv10 <- rsmp("cv",folds=10)
cv10$instantiate(task)
best_template <- L[[match(best_id,ids)]]$clone(deep=TRUE)
rr10 <- resample(task,best_template,cv10,store_models=FALSE)
oof10_pred <- pred_dt(rr10$prediction(),"10fold_OOF")
oof10_perf <- cbind(model=best_id,score(rr10$prediction()))
# Fit the selected learner on all participants for subsequent application.
best_model_all <- L[[match(best_id,ids)]]$clone(deep=TRUE)
best_model_all$train(task)
fwrite(train_perf,       "<TRAINING_CV_PERFORMANCE_CSV>")
fwrite(test_perf,        "<TEST_PERFORMANCE_CSV>")
fwrite(best_train_pred,  "<TRAINING_OOF_PREDICTIONS_CSV>")
fwrite(best_test_pred,   "<TEST_PREDICTIONS_CSV>")
fwrite(oof10_pred,       "<ALL_SUBJECTS_OOF_PREDICTIONS_CSV>")
fwrite(oof10_perf,       "<ALL_SUBJECTS_OOF_PERFORMANCE_CSV>")
fwrite(feature_map,      "<FEATURE_MAP_CSV>")
saveRDS(best_model,     "<TRAINED_MODEL_RDS>")

# -----------------------------------------------------------------------------
# 4. Feature importance and SHAP values
# -----------------------------------------------------------------------------
library(lightgbm)
anno=read.csv('<FEATURE_DICTIONARY_CSV>')
fm <- as.data.table(feature_map); an <- as.data.table(anno)
fm[, fid:=as.integer(sub("-.*","",original))]
an[, fid:=as.integer(sub("^p(\\d+).*","\\1",name))]
fa <- merge(fm, unique(an[,.(fid,title,type,coding_name,units,folder_path)]),
            by="fid", all.x=TRUE)
if(!exists("best_model")) best_model <- readRDS("<TRAINED_MODEL_RDS>")

imp <- as.data.table(lgb.importance(best_model$model, percentage=FALSE))
setnames(imp,c("Feature","Gain","Cover","Frequency"),c("feature","gain","cover","frequency"))
imp[, gain_pct:=gain/sum(gain)]
fi <- merge(fa,imp,by="feature",all.x=TRUE)
fi[is.na(gain),c("gain","cover","frequency","gain_pct"):=0]
setorder(fi,-gain)

#fwrite(fi,"<SHAP_IMPORTANCE_CSV>")

library(data.table); library(lightgbm)
m <- if(file.exists("<ALL_SUBJECTS_MODEL_RDS>")) readRDS("<ALL_SUBJECTS_MODEL_RDS>") else readRDS("<TRAINED_MODEL_RDS>")
X <- as.matrix(data[,feature_map$original,drop=FALSE]); colnames(X) <- feature_map$feature
S <- predict(m$model,X,type="contrib"); base <- S[,ncol(S)]; S <- S[,-ncol(S),drop=FALSE]; colnames(S) <- colnames(X)
shap_imp <- data.table(feature=colnames(X),mean_abs_shap=colMeans(abs(S)),
                       mean_shap=colMeans(S),direction_r=mapply(\(x,s) cor(x,s,method="spearman"),as.data.frame(X),as.data.frame(S)))
shap_imp <- merge(unique(fi[,.(feature,original,field_id=fid,title)]),shap_imp,by="feature",all.y=TRUE)
shap_imp[,direction:=fcase(direction_r>.05,"higher → older",direction_r< -.05,"higher → younger",default="weak/mixed")]
fwrite(shap_imp,"<SHAP_IMPORTANCE_CSV>")
setorder(shap_imp,-mean_abs_shap)

# -----------------------------------------------------------------------------
# 5. Psychological age acceleration and phenotype associations
# -----------------------------------------------------------------------------
phe=readRDS('<PHENOTYPE_DATA_RDS>')
anno=readRDS('<PHENOTYPE_ANNOTATIONS_RDS>')
cov=readRDS('<COVARIATES_RDS>')
cov=cov[,c('sex','ethnicity_5cat','town_index','qual_category','income','assessment_centre','employment','smoking')]
x <- as.character(cov$assessment_centre)
cov$assessment_centre <- factor(fcase(
  x%in%c("11004","11005"),"Scotland",
  x%in%c("11003","11022","11023"),"Wales",
  x%in%c("11009","11017","11027","11010","11014"),"NE/Yorkshire",
  x%in%c("11008","11001","11016","10003","11024","11025"),"North West",
  x%in%c("11021","11013","11006"),"Midlands",
  x%in%c("11011","11028","11002","11007","11026"),"South",
  x%in%c("11012","11020","11018"),"London"
))
setDT(phe); setDT(oof10_pred)
id <- intersect(oof10_pred$eid,phe$eid)
oof <- oof10_pred[eid %in% id]
p <- phe[match(oof$eid,phe$eid)]

str(oof10_pred)
str(cov)
str(p[,1:20])

library(data.table); library(parallel)
CORES <- min(50L,max(1L,detectCores()-1L))
covn <- names(cov); fac_cov <- covn[vapply(cov,is.factor,logical(1))]; lev <- lapply(cov[fac_cov],levels)

## ---------- PAA ----------
pa <- as.data.table(oof10_pred)[complete.cases(eid,pred,truth),
                                .(eid=as.character(eid),age=as.numeric(truth),pred=as.numeric(pred))]
stopifnot(!anyDuplicated(pa$eid))
pf <- lm(pred~age,pa); pa[,PAA:=resid(pf)][,PAA_z:=as.numeric(scale(PAA))]

## ---------- merge PAA + cov + phenotype ----------
rn <- rownames(cov)
if(is.null(rn)||identical(rn,as.character(seq_len(nrow(cov))))) stop("Covariate row names must contain participant IDs.")
cv <- as.data.table(copy(cov)); cv[,eid:=as.character(rn)]; setcolorder(cv,c("eid",covn))
pp <- as.data.table(copy(p)); pp[,eid:=as.character(eid)]
stopifnot(!anyDuplicated(cv$eid),!anyDuplicated(pp$eid))
pheno <- setdiff(names(pp),"eid")
if(length(intersect(pheno,covn))) stop("Phenotype and covariate names overlap.")

dat <- Reduce(function(x,y)merge(x,y,by="eid",all=FALSE),list(pa,cv,pp))
for(v in fac_cov) set(dat,j=v,value=factor(dat[[v]],levels=lev[[v]]))

## ---------- Official UK Biobank annotations ----------
fid <- suppressWarnings(as.integer(sub("^p([0-9]+).*$","\\1",pheno)))
if(anyNA(fid)) stop("Cannot extract UK Biobank field IDs: ",paste(pheno[is.na(fid)],collapse=", "))

fm <- fread("https://biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=1")
fm[,`:=`(field_id=as.integer(field_id),value_type=as.integer(value_type),encoding_id=as.integer(encoding_id))]
meta <- merge(data.table(var=pheno,field_id=fid),
              fm[,.(field_id,title,value_type,encoding_id,units)],by="field_id",all.x=TRUE)
if(anyNA(meta$value_type)) stop("Some field IDs were not found in UK Biobank schema 1.")
meta[,type:=fcase(value_type%in%c(11L,31L),"numeric",
                  value_type%in%c(21L,22L,41L),"factor",default="skip")]

enc <- rbindlist(lapply(c(5,7,11),function(i){
  x <- fread(sprintf("https://biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=%d",i))
  x[,.(encoding_id=as.integer(encoding_id),value=as.character(value),meaning=as.character(meaning))]
}),fill=TRUE)
miss_re <- "prefer not to answer|do not know|not known|not calculable|unable to answer|missing"

prep_pheno <- function(x,m){
  e <- if(is.na(m$encoding_id)) enc[0] else enc[encoding_id==m$encoding_id]
  if(m$type=="numeric"){
    z <- suppressWarnings(as.numeric(as.character(x)))
    if(nrow(e)){
      ev <- suppressWarnings(as.numeric(e$value))
      z[z%in%ev[grepl(miss_re,e$meaning,ignore.case=TRUE)]] <- NA_real_
      z[z%in%ev[grepl("^less than",e$meaning,ignore.case=TRUE)]] <- .5
    }
    if(m$field_id==20240L) z[z==-1] <- NA_real_
    return(z)
  }
  if(m$type=="factor"){
    z <- as.character(x)
    if(nrow(e)){
      z[z%in%e$value[grepl(miss_re,e$meaning,ignore.case=TRUE)]] <- NA_character_
      mp <- setNames(e$meaning,e$value); hit <- z%in%names(mp); z[hit] <- unname(mp[z[hit]])
    }
    return(factor(z))
  }
  rep(NA,length(x))
}

## ---------- Per-phenotype model: PAA ~ phenotype + age + covariates ----------
one <- function(v){
  setDTthreads(1); m <- meta[var==v][1]
  tryCatch({
    if(m$type=="skip") stop("The UK Biobank value type is not supported by this regression step.")

    ## Exclude missing phenotype values, then apply UK Biobank coding.
    q <- copy(dat[!is.na(get(v)),c("PAA","age",covn,v),with=FALSE]); setnames(q,v,"pheno")
    q[,pheno:=prep_pheno(pheno,m)]
    q <- q[!is.na(pheno)&complete.cases(q[,c("PAA","age",covn),with=FALSE])]

    ## Drop unused factor levels within this phenotype analysis sample.
    for(z in fac_cov) set(q,j=z,value=droplevels(q[[z]]))
    if(is.factor(q$pheno)) q[,pheno:=droplevels(pheno)]
    if((is.factor(q$pheno)&&nlevels(q$pheno)<2L)||(!is.factor(q$pheno)&&uniqueN(q$pheno)<2L))
      stop("The phenotype has no usable variation.")

    usecov <- covn[vapply(covn,function(z){
      u <- q[[z]]; if(is.factor(u)) nlevels(u)>1L else uniqueN(u)>1L
    },logical(1))]

    f <- lm(reformulate(c("pheno","age",usecov),response="PAA"),q)
    s <- summary(f); co <- s$coefficients; ii <- grep("^pheno",rownames(co))
    if(!length(ii)) stop("The phenotype coefficient cannot be estimated.")

    gp <- tryCatch(drop1(f,test="F")["pheno","Pr(>F)"],error=function(e) NA_real_)
    cr <- qt(.975,df.residual(f))

    data.table(var=v,field_id=m$field_id,title=m$title,type=m$type,units=m$units,
               N=nobs(f),levels=if(is.factor(q$pheno))nlevels(q$pheno) else uniqueN(q$pheno),
               reference=if(is.factor(q$pheno))levels(q$pheno)[1] else NA_character_,
               term=rownames(co)[ii],beta=co[ii,1],SE=co[ii,2],t=co[ii,3],p=co[ii,4],
               lower=co[ii,1]-cr*co[ii,2],upper=co[ii,1]+cr*co[ii,2],
               overall_p=gp,R2=s$r.squared,adj_R2=s$adj.r.squared,error=NA_character_)
  },error=function(e)data.table(var=v,field_id=m$field_id,title=m$title,type=m$type,
                                N=NA_integer_,error=conditionMessage(e)))
}

PAA_pheno <- rbindlist(mclapply(pheno,one,mc.cores=CORES,mc.preschedule=TRUE),fill=TRUE)

## Multiple-testing adjustment at the phenotype level
G <- unique(PAA_pheno[!is.na(overall_p),.(var,overall_p)])[order(overall_p)]
G[,FDR:=p.adjust(overall_p,"BH")]
PAA_pheno[,FDR:=NA_real_]; PAA_pheno[G,on="var",FDR:=i.FDR]
setorder(PAA_pheno,FDR,overall_p,p,na.last=TRUE)

RESULTS <- list(PAA_model=summary(pf),phenotype_meta=meta,PAA_phenotype=PAA_pheno)
saveRDS(RESULTS,'<PHENOTYPE_RESULTS_RDS>')

# -----------------------------------------------------------------------------
# 6. Incident disease, disease burden, and mortality
# -----------------------------------------------------------------------------
cov=readRDS('<COVARIATES_RDS>')
cov=cov[,c('sex','ethnicity_5cat','town_index','qual_category','income','assessment_centre','employment','smoking')]
x <- as.character(cov$assessment_centre)
cov$assessment_centre <- factor(fcase(
  x%in%c("11004","11005"),"Scotland",
  x%in%c("11003","11022","11023"),"Wales",
  x%in%c("11009","11017","11027","11010","11014"),"NE/Yorkshire",
  x%in%c("11008","11001","11016","10003","11024","11025"),"North West",
  x%in%c("11021","11013","11006"),"Midlands",
  x%in%c("11011","11028","11002","11007","11026"),"South",
  x%in%c("11012","11020","11018"),"London"
))

oof10_pred
age_data=datall$age_data_2016
dis=readRDS('<DISEASE_AND_DEATH_RDS>')$hosp_events
death=readRDS('<DISEASE_AND_DEATH_RDS>')$death_data
str(cov)
str(oof10_pred)
str(age_data)
str(dis)
str(death)

library(data.table); library(survival); library(MASS); library(cmprsk); library(parallel)

CORES <- min(50L,max(1L,detectCores()-1L)); MIN_EVENT <- 100L
HEND <- as.IDate("2022-05-31"); DEND_CAP <- as.IDate("2024-08-31")
Sys.setenv(OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1); setDTthreads(CORES)

covn <- c("sex","ethnicity_5cat","town_index","qual_category","income","assessment_centre","employment","smoking")
fac <- c("ethnicity_5cat","qual_category","assessment_centre","employment")
num <- c("sex","town_index","income","smoking")
primary_letter <- LETTERS[1:14]

capture_fit <- function(expr){
  w <- character()
  f <- withCallingHandlers(expr,warning=function(e){w<<-c(w,conditionMessage(e)); invokeRestart("muffleWarning")})
  list(fit=f,warning=if(length(w)) paste(unique(w),collapse=" | ") else NA_character_)
}
dropfac <- function(x){
  x <- copy(x)
  for(v in intersect(fac,names(x))) set(x,j=v,value=droplevels(x[[v]]))
  x
}
usable <- function(x,v) v[vapply(v,function(s){
  z <- x[[s]]
  if(is.factor(z)) nlevels(droplevels(z))>1L else length(unique(z[!is.na(z)]))>1L
},logical(1))]
rhs_for <- function(x,burden=FALSE){
  v <- usable(x,c("PAA_z","age",covn,if(burden)"prev_n"))
  if(!"PAA_z"%in%v) stop("PAA_z has no variation in this risk set.")
  paste(v,collapse="+")
}
mm_for <- function(x,burden=FALSE){
  x <- dropfac(x); v <- usable(x,c("PAA_z","age",covn,if(burden)"prev_n"))
  mm <- model.matrix(reformulate(v),as.data.frame(x))[,-1,drop=FALSE]
  keep <- vapply(seq_len(ncol(mm)),function(j) length(unique(mm[,j]))>1L,logical(1))
  mm <- mm[,keep,drop=FALSE]
  if(!"PAA_z"%in%colnames(mm)) stop("PAA_z is absent from the design matrix.")
  mm
}
pull_eff <- function(f){
  b <- unname(coef(f)["PAA_z"]); se <- sqrt(vcov(f)["PAA_z","PAA_z"])
  if(!is.finite(b)||!is.finite(se)||se<=0) stop("The PAA_z coefficient or standard error cannot be estimated.")
  c(est=exp(b),lower=exp(b-1.96*se),upper=exp(b+1.96*se),p=2*pnorm(-abs(b/se)))
}

## ---------- 1. PAA ----------
rn <- rownames(cov); cv <- as.data.table(copy(cov))
if(!"eid"%in%names(cv)){
  if(is.null(rn)||identical(rn,as.character(seq_len(nrow(cov))))) stop("Covariates have neither an eid column nor participant IDs in row names.")
  cv[,eid:=as.integer(rn)]
}else cv[,eid:=as.integer(eid)]
if(any(!covn%in%names(cv))) stop("Missing covariates: ",paste(setdiff(covn,names(cv)),collapse=", "))
for(v in fac) set(cv,j=v,value=factor(cv[[v]]))
lev <- setNames(lapply(fac,function(v) levels(cv[[v]])),fac)
for(v in num) set(cv,j=v,value=suppressWarnings(as.numeric(as.character(cv[[v]]))))

pr <- as.data.table(oof10_pred)[,.(eid=as.integer(eid),pred=as.numeric(pred))]
ag <- as.data.table(age_data)[mhq2016_completed==TRUE,.(eid=as.integer(eid),mhq=as.IDate(mhq2016_date),age=as.numeric(age_mhq2016))]
de <- as.data.table(death)[,.(eid=as.integer(eid),death_date=as.IDate(death_date))]
stopifnot(!anyDuplicated(cv$eid),!anyDuplicated(pr$eid),!anyDuplicated(ag$eid),!anyDuplicated(de$eid))
DEND_OBS <- de[!is.na(death_date),max(death_date)]
if(!length(DEND_OBS)||is.na(DEND_OBS)) stop("No valid death dates were found.")
DEND <- min(DEND_CAP,DEND_OBS)

pa <- merge(pr,ag,by="eid")[complete.cases(pred,age,mhq)]
pf <- lm(pred~age,pa); pa[,PAA:=resid(pf)][,PAA_z:=as.numeric(scale(PAA))]

am <- Reduce(function(x,y)merge(x,y,by="eid"),list(pa,cv,de))
for(v in fac) set(am,j=v,value=factor(as.character(am[[v]]),levels=lev[[v]]))
mis <- data.table(variable=c("PAA_z","age",covn),
                  n_missing=sapply(am[,c("PAA_z","age",covn),with=FALSE],function(x)sum(is.na(x))))
mis[,pct:=n_missing/nrow(am)]

a <- am[complete.cases(am[,c("PAA_z","age",covn),with=FALSE])&(is.na(death_date)|death_date>mhq)]
for(v in fac) set(a,j=v,value=droplevels(a[[v]]))
a[,censor:=HEND][!is.na(death_date)&death_date<censor,censor:=death_date]

s <- coef(summary(pf))["age",]
PAA_summary <- data.table(N_PAA=nrow(pa),N_analysis=nrow(a),beta_age=s[1],SE=s[2],p=s[4],
                          R2=summary(pf)$r.squared,PAA_SD=sd(pa$PAA),cor_PAA_age=cor(pa$PAA,pa$age))

## ---------- 2. ICD10 -> BLOCK ----------
H <- fread("https://biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=12")[encoding_id==19]
H[,`:=`(code_id=as.character(code_id),parent_id=as.character(parent_id),
        value_norm=toupper(gsub("\\.","",trimws(value))))]
clean_name <- function(x) trimws(sub("^[A-Z][0-9]{2}(?:\\.[A-Z0-9]+)?(?:-[A-Z]?[0-9]{2})?\\s+","",x,perl=TRUE))
BL <- H[startsWith(value,"Block "),.(code_id,block=sub("^Block ","",value),block_name=clean_name(meaning))]
P <- setNames(H$parent_id,H$code_id); BD <- setNames(BL$block,BL$code_id); BN <- setNames(BL$block_name,BL$code_id)

d0 <- as.data.table(copy(dis))[,`:=`(eid=as.integer(eid),icd10=toupper(gsub("\\.","",trimws(icd10))),event_date=as.IDate(event_date))]
d0[,icd3:=substr(icd10,1,3)]
M <- unique(d0[!is.na(icd10)&icd10!="",.(icd10)])
M[,node:=H$code_id[match(icd10,H$value_norm)]]
cur <- M$node; bid <- rep(NA_character_,nrow(M))
for(i in 1:30){
  z <- is.na(bid)&cur%in%BL$code_id; bid[z] <- cur[z]
  z <- is.na(bid)&!is.na(cur); if(!any(z)) break
  cur[z] <- unname(P[cur[z]])
}
M[,`:=`(icd_block=unname(BD[bid]),block_name=unname(BN[bid]))]
d0 <- M[d0,on="icd10"]

H3 <- unique(H[grepl("^[A-Z][0-9]{2}$",value_norm),.(code=value_norm,name=clean_name(meaning))],by="code")
N3 <- setNames(H3$name,H3$code); NB <- setNames(BL$block_name,BL$block)

## ---------- 3. Disease records: primary analysis restricted to ICD-10 A-N ----------
dd_all <- unique(d0[eid%in%a$eid&!is.na(event_date)&event_date<=HEND&icd3!="",
                    .(eid,icd10,icd3,icd_block,block_name,event_date)])
dd_all <- merge(dd_all,a[,.(eid,mhq,censor)],by="eid")[event_date<=censor]
dd <- dd_all[substr(icd3,1,1)%chin%primary_letter]

pre <- unique(dd[event_date<=mhq,.(eid,icd3)])[,.(prev_n=.N),by=eid]
a[,prev_n:=0L][pre,on="eid",prev_n:=i.prev_n]

rhs <- paste(c("PAA_z","age",covn),collapse="+")
rhsB <- paste(c("PAA_z","age",covn,"prev_n"),collapse="+")
rhs0 <- paste(c("age",covn),collapse="+")

## ---------- 4. incident endpoint ----------
prep <- function(v,lag=0){
  z <- unique(dd[!is.na(get(v))&get(v)!="",.(eid,code=get(v),event_date,mhq,censor)])
  days <- as.integer(round(365.25*lag)); z[,start:=mhq+days]; z <- z[start<censor]
  pv <- unique(z[event_date<=start,.(code,eid)])
  ev <- z[event_date>start&event_date<=censor]
  if(nrow(pv)) ev <- ev[!pv,on=.(code,eid)]
  ev <- ev[,.(event_date=min(event_date)),by=.(code,eid)]
  setkey(pv,code,eid); setkey(ev,code,eid)
  list(prev=pv,ev=ev,days=days)
}
P3 <- prep("icd3"); P32 <- prep("icd3",2); PB <- prep("icd_block")
D10 <- unique(dd[,.(eid,icd10,event_date,mhq,censor)])

## ---------- 5. Overall disease incidence and disease burden ----------
burden <- function(pp){
  x <- copy(a)[,start:=mhq+pp$days][censor>start]
  b <- pp$ev[,.(n_new_icd3=.N,first_date=min(event_date)),by=eid]
  q <- copy(D10)[,start:=mhq+pp$days][event_date>start&event_date<=censor,.(n_new_icd10=uniqueN(icd10)),by=eid]
  x[,`:=`(n_new_icd3=0L,n_new_icd10=0L,first_date=as.IDate(NA))]
  x[b,on="eid",`:=`(n_new_icd3=i.n_new_icd3,first_date=i.first_date)]
  x[q,on="eid",n_new_icd10:=i.n_new_icd10]
  x[,`:=`(event=as.integer(n_new_icd3>0),stop=censor)]
  x[!is.na(first_date),stop:=first_date]
  x[,`:=`(followup=as.numeric(censor-start)/365.25,time=as.numeric(stop-start)/365.25)]
  dropfac(x[followup>0&time>0])
}
E <- function(f,out,effect,model,lag,N,cases,total=NA_integer_,warning=NA_character_){
  e <- pull_eff(f)
  data.table(outcome=out,model,lag,N,cases,total,effect,estimate=e["est"],lower=e["lower"],upper=e["upper"],p=e["p"],warning)
}
fitburden <- function(x,lag){
  x <- dropfac(x); r <- rhs_for(x,FALSE); rb <- rhs_for(x,TRUE)
  f1 <- capture_fit(coxph(as.formula(paste("Surv(time,event)~",r)),x,ties="efron"))
  f2 <- capture_fit(coxph(as.formula(paste("Surv(time,event)~",rb)),x,ties="efron"))
  f3 <- capture_fit(glm.nb(as.formula(paste("n_new_icd3~",r,"+offset(log(followup))")),x))
  f4 <- capture_fit(glm.nb(as.formula(paste("n_new_icd3~",rb,"+offset(log(followup))")),x))
  f5 <- capture_fit(glm.nb(as.formula(paste("n_new_icd10~",r,"+offset(log(followup))")),x))
  rbindlist(list(
    E(f1$fit,"First incident ICD3","HR","main",lag,nrow(x),sum(x$event),warning=f1$warning),
    E(f2$fit,"First incident ICD3","HR","+baseline disease burden",lag,nrow(x),sum(x$event),warning=f2$warning),
    E(f3$fit,"Number of new ICD3","IRR","main",lag,nrow(x),sum(x$n_new_icd3>0),sum(x$n_new_icd3),f3$warning),
    E(f4$fit,"Number of new ICD3","IRR","+baseline disease burden",lag,nrow(x),sum(x$n_new_icd3>0),sum(x$n_new_icd3),f4$warning),
    E(f5$fit,"Number of new distinct ICD10 codes","IRR","main",lag,nrow(x),sum(x$n_new_icd10>0),sum(x$n_new_icd10),f5$warning)
  ))
}
b0 <- burden(P3); b2 <- burden(P32)
Overall <- rbindlist(list(fitburden(b0,0),fitburden(b2,2))); Overall[,FDR:=p.adjust(p,"BH")]

## ---------- 6. All-cause mortality ----------
mkdeath <- function(lag=0){
  days <- as.integer(round(365.25*lag))
  x <- copy(a)[,start:=mhq+days][start<DEND&(is.na(death_date)|death_date>start)]
  x[,`:=`(event=as.integer(!is.na(death_date)&death_date<=DEND),stop=DEND)]
  x[event==1L,stop:=death_date]
  x[,time:=as.numeric(stop-start)/365.25]
  dropfac(x[time>0])
}
fitdeath <- function(x,lag){
  r <- rhs_for(x,FALSE); rb <- rhs_for(x,TRUE)
  f1 <- capture_fit(coxph(as.formula(paste("Surv(time,event)~",r)),x,ties="efron"))
  f2 <- capture_fit(coxph(as.formula(paste("Surv(time,event)~",rb)),x,ties="efron"))
  rbindlist(list(
    E(f1$fit,"All-cause mortality","HR","main",lag,nrow(x),sum(x$event),warning=f1$warning),
    E(f2$fit,"All-cause mortality","HR","+baseline disease burden",lag,nrow(x),sum(x$event),warning=f2$warning)
  ))
}
m0 <- mkdeath(); m2 <- mkdeath(2)
Mortality <- rbindlist(list(fitdeath(m0,0),fitdeath(m2,2)))

## ---------- 7. Disease-specific Cox models: drop unused levels within each risk set ----------
risk <- function(k,pp){
  st <- as.integer(a$mhq)+pp$days; ce <- as.integer(a$censor)
  keep <- ce>st & !a$eid%in%pp$prev[.(k),eid]
  q <- copy(a[keep]); ev <- pp$ev[.(k)]; j <- match(q$eid,ev$eid); s <- !is.na(j)
  en <- as.integer(q$censor); en[s] <- as.integer(ev$event_date[j[s]])
  q[,`:=`(time=(en-(as.integer(mhq)+pp$days))/365.25,status=as.integer(s))]
  q <- q[time>0]
  sx <- unique(q[status==1,sex])
  if(length(sx)==1L && uniqueN(q$sex)>1L) q <- q[sex==sx]
  dropfac(q)
}
scan <- function(pp,level,nmap,min_event=MIN_EVENT,codes=NULL,burden=FALSE,model="main"){
  all <- is.null(codes); if(all) codes <- pp$ev[,.N,by=code][N>=min_event,code]
  if(!length(codes)) return(data.table())
  one <- function(k){
    setDTthreads(1); q <- risk(k,pp); N <- nrow(q); ne <- sum(q$status)
    nm <- unname(nmap[k]); if(is.na(nm)) nm <- ""
    tryCatch({
      r <- rhs_for(q,burden)
      fw <- capture_fit(coxph(as.formula(paste("Surv(time,status)~",r)),q,ties="efron",model=FALSE,x=FALSE,y=FALSE))
      e <- pull_eff(fw$fit)
      data.table(code=k,name=nm,N=N,events=ne,HR=e["est"],lower=e["lower"],upper=e["upper"],p=e["p"],
                 warning=fw$warning,error=NA_character_)
    },error=function(e)data.table(code=k,name=nm,N=N,events=ne,HR=NA_real_,lower=NA_real_,upper=NA_real_,
                                  p=NA_real_,warning=NA_character_,error=conditionMessage(e)))
  }
  r <- rbindlist(mclapply(codes,one,mc.cores=CORES,mc.preschedule=TRUE),fill=TRUE)
  r[,`:=`(FDR=if(all)p.adjust(p,"BH") else NA_real_,level=level,lag=pp$days/365.25,model=model)]
  r[order(FDR,p,na.last=TRUE)]
}

ICD3 <- scan(P3,"ICD3",N3)
ICD3_lag2 <- scan(P32,"ICD3",N3)
BLOCK <- scan(PB,"BLOCK",NB)
sig <- ICD3[!is.na(FDR)&FDR<.05&is.finite(HR),code]
ICD3_prevburden <- scan(P3,"ICD3",N3,codes=sig,burden=TRUE,model="+baseline disease burden")
ICD3_compare <- merge(
  ICD3[,.(code,name,N,events,HR,lower,upper,p,FDR,warning,error)],
  ICD3_lag2[,.(code,N_lag2=N,events_lag2=events,HR_lag2=HR,lower_lag2=lower,upper_lag2=upper,
               p_lag2=p,FDR_lag2=FDR,warning_lag2=warning,error_lag2=error)],by="code",all.x=TRUE)

## ---------- 8. Fine-Gray ----------
finegray <- function(codes){
  if(!length(codes)) return(data.table())
  one <- function(k){
    setDTthreads(1); nm <- unname(N3[k]); if(is.na(nm)) nm <- ""
    st <- as.integer(a$mhq); hend <- as.integer(HEND)
    keep <- hend>st & (is.na(a$death_date)|as.integer(a$death_date)>st) & !a$eid%in%P3$prev[.(k),eid]
    q <- dropfac(copy(a[keep])); status <- integer(nrow(q)); en <- rep(hend,nrow(q))
    d <- as.integer(q$death_date); z <- !is.na(d)&d<=hend; status[z] <- 2L; en[z] <- d[z]
    ev <- P3$ev[.(k)]; j <- match(q$eid,ev$eid); z <- !is.na(j); status[z] <- 1L; en[z] <- as.integer(ev$event_date[j[z]])
    sx <- unique(q$sex[status==1L])
    if(length(sx)==1L && uniqueN(q$sex)>1L){
      kk <- q$sex==sx; q <- q[kk]; status <- status[kk]; en <- en[kk]
    }
    N <- nrow(q); ne <- sum(status==1L); nd <- sum(status==2L)
    tryCatch({
      xx <- mm_for(q,FALSE); jPAA <- match("PAA_z",colnames(xx))
      fw <- capture_fit(crr((en-as.integer(q$mhq))/365.25,status,xx,failcode=1,cencode=0))
      b <- fw$fit$coef[jPAA]; se <- sqrt(fw$fit$var[jPAA,jPAA])
      if(!is.finite(b)||!is.finite(se)||se<=0) stop("The PAA_z subdistribution hazard ratio cannot be estimated.")
      data.table(code=k,name=nm,N=N,events=ne,deaths=nd,sHR=exp(b),lower=exp(b-1.96*se),
                 upper=exp(b+1.96*se),p=2*pnorm(-abs(b/se)),warning=fw$warning,error=NA_character_)
    },error=function(e)data.table(code=k,name=nm,N=N,events=ne,deaths=nd,sHR=NA_real_,lower=NA_real_,
                                  upper=NA_real_,p=NA_real_,warning=NA_character_,error=conditionMessage(e)))
  }
  rbindlist(mclapply(codes,one,mc.cores=CORES,mc.preschedule=TRUE),fill=TRUE)[order(p)]
}
FineGray <- finegray(sig)

# -----------------------------------------------------------------------------
# 7. Model diagnostics and nonlinearity
# -----------------------------------------------------------------------------
## ---------- 9. Proportional hazards and nonlinearity ----------
b0 <- dropfac(b0); m0 <- dropfac(m0)
rD <- rhs_for(b0,FALSE); rM <- rhs_for(m0,FALSE)
fD <- capture_fit(coxph(as.formula(paste("Surv(time,event)~",rD)),b0,x=TRUE,ties="efron"))$fit
fM <- capture_fit(coxph(as.formula(paste("Surv(time,event)~",rM)),m0,x=TRUE,ties="efron"))$fit
PH <- function(f,out){
  z <- cox.zph(f)$table
  data.table(outcome=out,term=rownames(z),chisq=z[,1],df=z[,2],p=z[,3])
}
PH_overall <- rbindlist(list(PH(fD,"First incident ICD3"),PH(fM,"All-cause mortality")))

phscan <- function(codes,pp,nmap){
  if(!length(codes)) return(data.table())
  one <- function(k) tryCatch({
    setDTthreads(1); q <- risk(k,pp); r <- rhs_for(q,FALSE)
    f <- capture_fit(coxph(as.formula(paste("Surv(time,status)~",r)),q,x=TRUE,ties="efron"))$fit
    z <- cox.zph(f)$table; nm <- unname(nmap[k]); if(is.na(nm)) nm <- ""
    data.table(code=k,name=nm,PH_PAA=z["PAA_z","p"],PH_global=z["GLOBAL","p"])
  },error=function(e){
    nm <- unname(nmap[k]); if(is.na(nm)) nm <- ""
    data.table(code=k,name=nm,PH_PAA=NA_real_,PH_global=NA_real_)
  })
  rbindlist(mclapply(codes,one,mc.cores=CORES,mc.preschedule=TRUE))
}
PH_ICD3 <- phscan(sig,P3,N3)
PH_BLOCK <- phscan(BLOCK[!is.na(FDR)&FDR<.05,code],PB,NB)

LRT <- function(f0,f1,out){
  x <- logLik(f0); y <- logLik(f1); LR <- 2*(as.numeric(y)-as.numeric(x)); df <- attr(y,"df")-attr(x,"df")
  data.table(outcome=out,LR,df,p=pchisq(LR,df,lower.tail=FALSE))
}
spline_fit <- function(x){
  x <- dropfac(x); v <- setdiff(usable(x,c("PAA_z","age",covn)),"PAA_z")
  coxph(as.formula(paste("Surv(time,event)~splines::ns(PAA_z,3)+",paste(v,collapse="+"))),x,ties="efron")
}
Nonlinearity <- rbindlist(list(LRT(fD,spline_fit(b0),"First incident ICD3"),
                               LRT(fM,spline_fit(m0),"All-cause mortality")))

## ---------- 10. QC + RESULTS ----------
Censor_QC <- data.table(metric=c("HEND","DEND_cap","last_observed_death","DEND_used"),
                        value=as.character(c(HEND,DEND_CAP,DEND_OBS,DEND)))
Model_QC <- data.table(
  result=c("ICD3","ICD3_lag2","BLOCK","ICD3_prevburden","FineGray"),
  n=c(nrow(ICD3),nrow(ICD3_lag2),nrow(BLOCK),nrow(ICD3_prevburden),nrow(FineGray)),
  failed=c(sum(is.na(ICD3$HR)),sum(is.na(ICD3_lag2$HR)),sum(is.na(BLOCK$HR)),
           sum(is.na(ICD3_prevburden$HR)),sum(is.na(FineGray$sHR))),
  warned=c(sum(!is.na(ICD3$warning)),sum(!is.na(ICD3_lag2$warning)),sum(!is.na(BLOCK$warning)),
           sum(!is.na(ICD3_prevburden$warning)),sum(!is.na(FineGray$warning)))
)
QC <- data.table(
  metric=c("OOF_N","PAA_N","analysis_N","disease_records_all","disease_records_A_N","ICD10_codes_A_N",
           "ICD3_codes_A_N","BLOCKs_A_N","unmapped_ICD10_codes"),
  value=c(nrow(pr),nrow(pa),nrow(a),nrow(dd_all),nrow(dd),uniqueN(dd$icd10),uniqueN(dd$icd3),
          uniqueN(dd$icd_block,na.rm=TRUE),M[is.na(icd_block),.N])
)
PAA_values <- a[,.(eid,mhq,age,pred,PAA,PAA_z,prev_n,death_date,censor)]

RESULTS <- list(PAA_summary=PAA_summary,QC=QC,Censor_QC=Censor_QC,Model_QC=Model_QC,Missing=mis,
                Overall=Overall,Mortality=Mortality,ICD3=ICD3,ICD3_lag2=ICD3_lag2,
                ICD3_compare=ICD3_compare,ICD3_prevburden=ICD3_prevburden,BLOCK=BLOCK,
                FineGray=FineGray,PH_overall=PH_overall,PH_ICD3=PH_ICD3,PH_BLOCK=PH_BLOCK,
                Nonlinearity=Nonlinearity,PAA_values=PAA_values)
saveRDS(RESULTS,'<DISEASE_MORTALITY_RESULTS_RDS>')

library(data.table); library(rms); library(survival)

dd<-rms::datadist(m0); dd$limits["Adjust to","PAA_z"]<-0; options(datadist="dd"); rcs<-rms::rcs
fit_rcs<-rms::cph(as.formula(paste0("survival::Surv(time,event) ~ rcs(PAA_z,4) + ",rhs0)),data=m0,x=TRUE,y=TRUE,surv=TRUE)

xs<-seq(quantile(m0$PAA_z,.01,na.rm=TRUE),quantile(m0$PAA_z,.99,na.rm=TRUE),length.out=300)
R<-as.data.table(rms::Predict(fit_rcs,PAA_z=xs,ref.zero=TRUE,fun=exp))

library(data.table); library(rms); library(survival)

dd<-rms::datadist(m0); dd$limits["Adjust to","PAA_z"]<-0; options(datadist="dd"); rcs<-rms::rcs
fit_rcs<-rms::cph(as.formula(paste0("survival::Surv(time,event) ~ rcs(PAA_z,4) + ",rhs0)),data=m0,x=TRUE,y=TRUE,surv=TRUE)
xs<-seq(quantile(m0$PAA_z,.01,na.rm=T),quantile(m0$PAA_z,.99,na.rm=T),length.out=300)
R<-as.data.table(rms::Predict(fit_rcs,PAA_z=xs,ref.zero=TRUE,fun=exp))
