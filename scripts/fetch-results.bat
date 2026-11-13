@echo off
setlocal
REM Pull /shared/ruler/results/hybridkv into .\results\ via a short Job that
REM streams a base64 tarball through its log.  No idle pods, no sleep.
cd /d "%~dp0.."

kubectl delete job hybridkv-export --ignore-not-found >nul 2>&1
kubectl apply -f k8s\job-export-results.yaml >nul
if errorlevel 1 goto fail
kubectl wait --for=condition=complete job/hybridkv-export --timeout=600s
if errorlevel 1 goto fail

if exist hybridkv-results.b64 del hybridkv-results.b64
if exist hybridkv-results.tar.gz del hybridkv-results.tar.gz
kubectl logs job/hybridkv-export --tail=-1 > hybridkv-results.b64
if errorlevel 1 goto fail
certutil -decode hybridkv-results.b64 hybridkv-results.tar.gz >nul
if errorlevel 1 goto fail
if not exist results mkdir results
tar -xzf hybridkv-results.tar.gz -C results
if errorlevel 1 goto fail
del hybridkv-results.b64
kubectl delete job hybridkv-export --ignore-not-found >nul 2>&1
echo Results in .\results\hybridkv
echo Analyse with: python analyze_hybrid.py results\hybridkv
exit /b 0

:fail
echo Export failed.
kubectl logs job/hybridkv-export --tail=20
exit /b 1
