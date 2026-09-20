$ErrorActionPreference = 'SilentlyContinue'
$root = $PSScriptRoot
$data = Join-Path $root 'data'
$cats = @('bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper')
$log  = Join-Path $root 'fix_result.txt'
$out = @()

foreach ($c in $cats) {
    $p = Join-Path $data $c
    if (Test-Path -LiteralPath $p) {
        takeown /f $p /r /d y | Out-Null
        icacls $p /grant '*S-1-1-0:(OI)(CI)F' /t /c /q | Out-Null
        attrib -r -s -h $p /s /d | Out-Null
        Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $p) { $out += "FAILED  $c" } else { $out += "DELETED $c" }
    } else {
        $out += "MISSING $c"
    }
}

$out | Out-File -LiteralPath $log -Encoding utf8
