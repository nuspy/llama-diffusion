<#
  build_engine.ps1 — compila i target diffusione del fork llama.cpp (PR #24423) con CUDA.

  Lezioni cristallizzate (non ovvie) per questa workstation:
    1. CUDA 13.1 rifiuta MSVC di VS18/2026 (toolset 14.51): nvcc accetta solo fino a VS2022.
       -> forziamo il toolset 14.44 con vcvars64.bat -vcvars_ver=14.44 (gia' installato sotto VS18).
    2. La variabile d'ambiente _CL_=/std:c++20 (presente in questa sessione) viene applicata da
       cl.exe a OGNI compilazione: rompe i file .c (c11+c++20 incompatibili) e fa crashare
       cudafe++ (0xC0000409) sui .cu. -> la azzeriamo prima di compilare.
    3. Generatore Ninja (niente integrazione CUDA-MSBuild su VS18); arch sm_120 per la RTX 5090.
    4. MSVC 14.44 ha default C++14: senza /std:c++17 i .cpp del core falliscono (C2429 structured
       bindings). -> forziamo -DCMAKE_CXX_STANDARD=17 (si applica solo ai .cpp, non ai .c).

  Uso:
    pwsh scripts/build_engine.ps1            # configure (se serve) + build dei target server+cli
    pwsh scripts/build_engine.ps1 -Clean     # ricrea la build dir da zero
#>
param(
    [switch]$Clean,
    [string[]]$Targets = @("llama-diffusion-gemma-visual-server", "llama-diffusion-cli"),
    [string]$VcVarsVer = "14.44",
    [string]$CudaArch  = "120"
)

$ErrorActionPreference = "Stop"
$Root   = Split-Path -Parent $PSScriptRoot
$Engine = Join-Path $Root "engine\llama.cpp"
$Build  = Join-Path $Engine "build"
$VcVars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path $VcVars)) { throw "vcvars64.bat non trovato: $VcVars" }
if (-not (Test-Path $Engine)) { throw "fork llama.cpp assente: $Engine (clona prima la PR #24423)" }

if ($Clean -and (Test-Path $Build)) { Remove-Item -Recurse -Force $Build }

# azzera le env var che cl.exe inietterebbe in ogni compilazione
Remove-Item Env:\_CL_ -ErrorAction SilentlyContinue
Remove-Item Env:\CL   -ErrorAction SilentlyContinue

$targetArgs = ($Targets | ForEach-Object { $_ }) -join " "

# cmake configure e' idempotente: lo eseguiamo sempre (riusa la cache, applica eventuali opzioni nuove)
$configure = "cmake -S `"$Engine`" -B `"$Build`" -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CudaArch " +
    "-DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release " +
    "-DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON && "

$cmd = "`"$VcVars`" -vcvars_ver=$VcVarsVer >nul && set `"_CL_=`" && set `"CL=`" && " +
       "$configure cmake --build `"$Build`" --target $targetArgs -j"

Write-Host ">> build con toolset $VcVarsVer, arch sm_$CudaArch, _CL_ azzerata" -ForegroundColor Cyan
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "build fallita (exit $LASTEXITCODE)" }

Write-Host ">> OK. Binari in: $Build\bin" -ForegroundColor Green
Get-ChildItem (Join-Path $Build "bin") -Filter "llama-diffusion*.exe" -ErrorAction SilentlyContinue |
    Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}
