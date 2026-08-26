param(
    [ValidateSet('default', 'fresh-shaders', 'query-lod', 'trace-cube', 'sync-compute', 'trace-compute', 'raw-rt', 'rt-stats', 'resident-off', 'resident-vb-off', 'resident-cbuf-off', 'fresh-cbuf', 'trace-gbuffer', 'trace-frame', 'trace-target-cbuf', 'dump-producer', 'inspect-producer', 'fermi-lease', 'memo-bypass', 'trace-producer')]
    [string]$Mode = 'default',
    [string]$Root = (Join-Path $PSScriptRoot '..'),
    [string]$Rom = 'C:\Users\Mythrax\Downloads\Dump\Super Mario Odyssey [0100000000010000][v0].dnsp',
    [int]$SampleSlot = -1,
    [int]$SampleComponent = -1,
    [int]$TexcoordSlot = -1,
    [string]$FsDebugTarget = '0x402fb0730',
    [int]$DebugOutputLoc = -1,
    [double]$ForceSampleLod = [double]::NaN,
    [string]$MaskMrtFsTexture = '',
    [string]$TextureMemoBypassVa = '',
    [string]$SkipDrawVs = '',
    [string]$NoBlendFs = '',
    [int]$ForceOutputLoc = -1,
    [string]$ForceOutputValue = '',
    [string]$ProbeShadeFs = '',
    [string]$VertexInputTrace = '',
    [string]$IrOctNormal = '',
    [switch]$SpirvFtz,
    [switch]$SignedZeroPreserve,
    [switch]$LegacyGuiUpload,
    [switch]$SyncRender,
    [int]$LateLoops = 20
)

$ErrorActionPreference = 'Stop'

$rom = (Resolve-Path -LiteralPath $Rom).Path
$work = (Resolve-Path $Root).Path
$exe = (Resolve-Path (Join-Path $work 'target\release\nexium.exe')).Path
$runId = [guid]::NewGuid().ToString('N')
$runStarted = Get-Date
$trigger = Join-Path $env:TEMP ('nexium-draw-trigger-' + $runId + '.flag')
$capture = Join-Path $env:TEMP ('nexium-smo-' + $Mode + '-' + $runId + '.png')
$logs = 'C:\Users\Mythrax\AppData\Roaming\NeXium\logs'
$before = Get-ChildItem -LiteralPath $logs -Filter 'nexium-*.log' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

$psi = [Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $exe
$psi.WorkingDirectory = $work
$psi.UseShellExecute = $false
$psi.Arguments = '"' + $rom + '"'
foreach ($key in @($psi.Environment.Keys)) {
    if ($key -like 'NEXIUM_*') {
        $psi.Environment.Remove($key)
    }
}
$psi.Environment['NEXIUM_LOG_LEVEL'] = 'info'
$psi.Environment['NEXIUM_BIND_TRACE_FS'] = '0x402fb0730,0x403ac0130'
$psi.Environment['NEXIUM_RT_ALIAS_UNIQUE'] = '1'
$psi.Environment['NEXIUM_RT_ALIAS_LIMIT'] = '1000'
$psi.Environment['NEXIUM_RT_ALIAS_META_DBG'] = '1'
$psi.Environment['NEXIUM_RT_PREFLIGHT_DBG'] = '1'
$psi.Environment['NEXIUM_PRESENT_DUMP'] = '1'
$psi.Environment['NEXIUM_PRESENT_DUMP_EVERY'] = '180'
$psi.Environment['NEXIUM_TEST_AUTOPRESS_A'] = '1'
if ($MaskMrtFsTexture) {
    $psi.Environment['NEXIUM_MASK_MRT_FS_TEXTURE'] = $MaskMrtFsTexture
}
if ($TextureMemoBypassVa) {
    $psi.Environment['NEXIUM_TEX_MEMO_BYPASS_VA'] = $TextureMemoBypassVa
}
if ($SkipDrawVs) {
    $psi.Environment['NEXIUM_SKIP_DRAW_VS'] = $SkipDrawVs
}
if ($NoBlendFs) {
    $psi.Environment['NEXIUM_NO_BLEND_FS'] = $NoBlendFs
}
if ($ForceOutputLoc -ge 0) {
    $psi.Environment['NEXIUM_NO_BUNDLE_CACHE'] = '1'
    $psi.Environment['NEXIUM_FS_DEBUG_TARGET'] = $FsDebugTarget
    $psi.Environment['NEXIUM_FS_FORCE_OUTPUT_LOC'] = [string]$ForceOutputLoc
    $psi.Environment['NEXIUM_FS_FORCE_OUTPUT_VALUE'] = $ForceOutputValue
}
if ($ProbeShadeFs) {
    $psi.Environment['NEXIUM_NO_BUNDLE_CACHE'] = '1'
    $psi.Environment['NEXIUM_PROBE_SHADE'] = '1'
    $psi.Environment['NEXIUM_PROBE_SHADE_FS'] = $ProbeShadeFs
    $psi.Environment['NEXIUM_PROBE_SHADE_COUNT'] = '0'
    $psi.Environment['NEXIUM_PROBE_SHADE_DIR'] = $env:TEMP
}
if ($VertexInputTrace) {
    $psi.Environment['NEXIUM_VERTEX_INPUT_TRACE'] = $VertexInputTrace
    $psi.Environment['NEXIUM_VERTEX_INPUT_TRACE_LIMIT'] = '64'
}
if ($IrOctNormal) {
    $psi.Environment['NEXIUM_NO_BUNDLE_CACHE'] = '1'
    $psi.Environment['NEXIUM_FS_DEBUG_TARGET'] = $FsDebugTarget
    $psi.Environment['NEXIUM_FS_DEBUG_OUTPUT_LOC'] = if ($DebugOutputLoc -ge 0) { [string]$DebugOutputLoc } else { '0' }
    $psi.Environment['NEXIUM_FS_IR_OCT_NORMAL'] = $IrOctNormal
}
if ($SpirvFtz) {
    $psi.Environment['NEXIUM_NO_BUNDLE_CACHE'] = '1'
    $psi.Environment['NEXIUM_SPIRV_FTZ'] = '1'
}
if ($SignedZeroPreserve) {
    $psi.Environment['NEXIUM_NO_BUNDLE_CACHE'] = '1'
    $psi.Environment['NEXIUM_SPIRV_SIGNED_ZERO_PRESERVE'] = '1'
}
if ($LegacyGuiUpload) {
    $psi.Environment['NEXIUM_LEGACY_GUI_UPLOAD'] = '1'
}
if ($SyncRender) {
    $psi.Environment['NEXIUM_ASYNC_RENDER'] = '0'
}
if ($SampleSlot -ge 0 -or $TexcoordSlot -ge 0 -or -not [double]::IsNaN($ForceSampleLod)) {
    $psi.Environment['NEXIUM_FS_DEBUG_TARGET'] = $FsDebugTarget
    if ($DebugOutputLoc -ge 0) {
        $psi.Environment['NEXIUM_FS_DEBUG_OUTPUT_LOC'] = [string]$DebugOutputLoc
    }
}
if (-not [double]::IsNaN($ForceSampleLod)) {
    $psi.Environment['NEXIUM_NO_BUNDLE_CACHE'] = '1'
    $psi.Environment['NEXIUM_FS_FORCE_SAMPLE_LOD'] = [string]$ForceSampleLod
}
if ($SampleSlot -ge 0) {
    $psi.Environment['NEXIUM_FS_SAMPLE_SLOT'] = [string]$SampleSlot
    if ($SampleComponent -ge 0) {
        $psi.Environment['NEXIUM_FS_SAMPLE_COMPONENT'] = [string]$SampleComponent
    }
}
if ($TexcoordSlot -ge 0) {
    $psi.Environment['NEXIUM_FS_TEXCOORD_SLOT'] = [string]$TexcoordSlot
}
if ($Mode -eq 'resident-off') {
    $psi.Environment['NEXIUM_RESIDENT_VB'] = '0'
    $psi.Environment['NEXIUM_RESIDENT_CBUF'] = '0'
}
if ($Mode -eq 'fresh-shaders') {
    $psi.Environment['NEXIUM_NO_BUNDLE_CACHE'] = '1'
}
if ($Mode -eq 'query-lod') {
    $psi.Environment['NEXIUM_NO_BUNDLE_CACHE'] = '1'
    $psi.Environment['NEXIUM_FS_DEBUG_TARGET'] = '0x405d90730'
    $psi.Environment['NEXIUM_FS_DEBUG_OUTPUT_LOC'] = '0'
    $psi.Environment['NEXIUM_FS_QUERY_LOD_SLOT'] = '0'
}
if ($Mode -eq 'trace-cube') {
    $psi.Environment['NEXIUM_NO_BUNDLE_CACHE'] = '1'
    $psi.Environment['NEXIUM_CUBE_RT_SYNC_TRACE'] = '1'
    $psi.Environment['NEXIUM_SMALL_RT_GATE_TRACE'] = '1'
}
if ($Mode -eq 'resident-vb-off') {
    $psi.Environment['NEXIUM_RESIDENT_VB'] = '0'
    $psi.Environment['NEXIUM_RESIDENT_VB_FORCE_FALLBACK'] = '1'
}
if ($Mode -eq 'resident-cbuf-off') {
    $psi.Environment['NEXIUM_RESIDENT_CBUF'] = '0'
}
if ($Mode -eq 'fresh-cbuf') {
    $psi.Environment['NEXIUM_CBUF_FRESH_READ'] = '1'
}
if ($Mode -eq 'trace-gbuffer') {
    $psi.Environment['NEXIUM_DRAW_TRACE'] = '1'
    $psi.Environment['NEXIUM_DRAW_TRACE_RT_VA'] = '0x523dd0000,0x5247f0000,0x523460000,0x523e80000'
    $psi.Environment['NEXIUM_DRAW_TRACE_MIN_V'] = '100'
    $psi.Environment['NEXIUM_DRAW_TRACE_END'] = '1200'
}
if ($Mode -eq 'trace-frame') {
    $psi.Environment['NEXIUM_DRAW_TRACE'] = '1'
    $psi.Environment['NEXIUM_DRAW_TRACE_TRIGGER_FILE'] = $trigger
    $psi.Environment['NEXIUM_DRAW_TRACE_WIDTH'] = '1600'
    $psi.Environment['NEXIUM_DRAW_TRACE_HEIGHT'] = '900'
    $psi.Environment['NEXIUM_DRAW_TRACE_END'] = '1200'
}
if ($Mode -eq 'trace-target-cbuf') {
    $psi.Environment['NEXIUM_NO_BUNDLE_CACHE'] = '1'
    $psi.Environment['NEXIUM_DRAW_TRACE'] = '1'
    $psi.Environment['NEXIUM_DRAW_TRACE_FS'] = '0x405d90730'
    $psi.Environment['NEXIUM_DRAW_TRACE_WIDTH'] = '1600'
    $psi.Environment['NEXIUM_DRAW_TRACE_HEIGHT'] = '900'
    $psi.Environment['NEXIUM_DRAW_TRACE_MIN_V'] = '100'
    $psi.Environment['NEXIUM_DRAW_TRACE_END'] = '20'
    $psi.Environment['NEXIUM_DRAW_TRACE_CBUF_FULL'] = '1'
    $psi.Environment['NEXIUM_DRAW_TRACE_CBUF_MAX'] = '0x180'
    $psi.Environment['NEXIUM_PROBE_SHADE'] = '1'
    $psi.Environment['NEXIUM_PROBE_SHADE_FS'] = '0x405d90730'
    $psi.Environment['NEXIUM_PROBE_SHADE_COUNT'] = '0'
    $psi.Environment['NEXIUM_PROBE_SHADE_DIR'] = $env:TEMP
}
if ($Mode -eq 'dump-producer') {
    $psi.Environment['NEXIUM_DUMP_VS'] = '0x405d90030'
    $psi.Environment['NEXIUM_DUMP_FS'] = '0x405d90730'
    $psi.Environment['NEXIUM_SHADER_MAP_DBG'] = '1'
    $psi.Environment['NEXIUM_MIP_UPLOAD_STATS'] = '1'
    $psi.Environment['NEXIUM_TEXDUMP'] = '1'
    $psi.Environment['NEXIUM_TEXDUMP_IMG'] = '1'
    $psi.Environment['NEXIUM_TEXDUMP_VA'] = '0x594080000'
}
if ($Mode -eq 'inspect-producer') {
    $psi.Environment['NEXIUM_NO_BUNDLE_CACHE'] = '1'
    $psi.Environment['NEXIUM_RESIDENT_VB'] = '0'
    $psi.Environment['NEXIUM_BIND_TRACE_FS'] = '0x405d90730'
    $psi.Environment['NEXIUM_MENU_DRAW_DBG'] = '1'
    $psi.Environment['NEXIUM_MENU_DRAW_NVMAP'] = 'all'
    $psi.Environment['NEXIUM_MENU_DRAW_FS'] = '0x405d90730'
    $psi.Environment['NEXIUM_MENU_DRAW_LIMIT'] = '30'
    $psi.Environment['NEXIUM_DRAW_TRACE_ATTR_LIMIT'] = '31'
    $psi.Environment['NEXIUM_VERTEX_INPUT_TRACE'] = '0x405d90030,0x405d90730,4'
    $psi.Environment['NEXIUM_VERTEX_INPUT_TRACE_LIMIT'] = '8'
    $psi.Environment['NEXIUM_TEXDUMP'] = '1'
    $psi.Environment['NEXIUM_TEXDUMP_IMG'] = '1'
    $psi.Environment['NEXIUM_TEXDUMP_VA'] = '0x574105c00'
    $psi.Environment['NEXIUM_DUMP_VS'] = '0x405d90030'
    $psi.Environment['NEXIUM_DUMP_FS'] = '0x405d90730'
    $psi.Environment['NEXIUM_SHADER_MAP_DBG'] = '1'
}
if ($Mode -eq 'fermi-lease') {
    $psi.Environment['NEXIUM_FERMI_RT_SNAPSHOT_LEASE'] = '1'
}
if ($Mode -eq 'memo-bypass') {
    $psi.Environment['NEXIUM_TEX_MEMO_BYPASS_VA'] = '0x594080000'
    $psi.Environment['NEXIUM_TEXDUMP'] = '1'
    $psi.Environment['NEXIUM_TEXDUMP_IMG'] = '1'
    $psi.Environment['NEXIUM_TEXDUMP_VA'] = '0x594080000'
}
if ($Mode -eq 'trace-producer') {
    $psi.Environment['NEXIUM_WATCH_WRITE_GPU'] = '0x594080000:0x40000'
    $psi.Environment['NEXIUM_WATCH_PAGE_PROTECT'] = '1'
    $psi.Environment['NEXIUM_WATCH_WRITE_LIMIT'] = '64'
    $psi.Environment['NEXIUM_WATCH_KEEP_AFTER_HIT'] = '1'
    $psi.Environment['NEXIUM_GUEST_PROBE'] = '0x594080000:0x40'
    $psi.Environment['NEXIUM_COMPUTE_RESOURCE_TRACE'] = '1'
}
if ($Mode -eq 'sync-compute') {
    $psi.Environment['NEXIUM_NO_LAZY_COMPUTE'] = '1'
}
if ($Mode -eq 'trace-compute') {
    $psi.Environment['NEXIUM_COMPUTE_RESOURCE_TRACE'] = '1'
    $psi.Environment['NEXIUM_COMPUTE_ALIAS_TRACE'] = '1'
}
if ($Mode -eq 'raw-rt') {
    $psi.Environment['NEXIUM_RAW_RT_TARGET'] = '0x5046a8200'
    $psi.Environment['NEXIUM_SHADER_MAP_DBG'] = '1'
    $psi.Environment['NEXIUM_COMPUTE_RESOURCE_TRACE'] = '1'
    $psi.Environment['NEXIUM_COMPUTE_ALIAS_TRACE'] = '1'
}
if ($Mode -eq 'rt-stats') {
    $psi.Environment['NEXIUM_RT_STATS'] = '1'
    $psi.Environment['NEXIUM_RT_DUMP'] = '1'
    $psi.Environment['NEXIUM_RT_STATS_START'] = '300'
    $psi.Environment['NEXIUM_RT_STATS_PERIOD'] = '300'
    $psi.Environment['NEXIUM_RT_STATS_MAX'] = '0'
    $psi.Environment['NEXIUM_RT_STATS_KEYS'] = '12:1600x900@0x524a70000,12:1600x900@0x523460000,12:1600x900@0x5046a8200,12:1600x900@0x523dd0000,12:1600x900@0x5257c0000,12:1600x900@0x5241b0000,12:1600x900@0x5053f8e00,12:1600x900@0x524b20000,12:1600x900@0x525c30000,12:1600x900@0x524620000,12:1600x900@0x505869e00,12:1600x900@0x524f90000'
}

$p = [Diagnostics.Process]::Start($psi)

Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
public static class NxCodexTraceWindow {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);
    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern bool SetWindowPos(IntPtr hWnd, IntPtr after, int x, int y, int cx, int cy, uint flags);
    [DllImport("user32.dll")]
    public static extern bool PostMessage(IntPtr hWnd, uint msg, IntPtr wp, IntPtr lp);
    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out Rect rect);
    [DllImport("user32.dll")]
    public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdc, uint flags);
    [StructLayout(LayoutKind.Sequential)]
    public struct Rect { public int Left, Top, Right, Bottom; }
    public static IntPtr[] ProcessWindows(uint processId) {
        var windows = new List<IntPtr>();
        EnumWindows(delegate(IntPtr hWnd, IntPtr lParam) {
            uint owner;
            GetWindowThreadProcessId(hWnd, out owner);
            if (owner == processId && IsWindowVisible(hWnd)) {
                windows.Add(hWnd);
            }
            return true;
        }, IntPtr.Zero);
        return windows.ToArray();
    }
}
'@

Add-Type -AssemblyName System.Drawing

function Move-NeXiumToBottom {
    param([Diagnostics.Process]$Process)
    $Process.Refresh()
    $handle = [IntPtr]::Zero
    $largestArea = 0L
    foreach ($candidate in [NxCodexTraceWindow]::ProcessWindows([uint32]$Process.Id)) {
        $rect = [NxCodexTraceWindow+Rect]::new()
        if ([NxCodexTraceWindow]::GetWindowRect($candidate, [ref]$rect)) {
            $area = [int64]($rect.Right - $rect.Left) * [int64]($rect.Bottom - $rect.Top)
            if ($area -gt $largestArea) {
                $largestArea = $area
                $handle = $candidate
            }
        }
        [void][NxCodexTraceWindow]::SetWindowPos($candidate, [IntPtr]1, 0, 0, 0, 0, 0x0013)
    }
    return $handle
}

$hwnd = [IntPtr]::Zero
for ($i = 0; $i -lt 100 -and $hwnd -eq [IntPtr]::Zero; $i++) {
    Start-Sleep -Milliseconds 100
    $hwnd = Move-NeXiumToBottom -Process $p
}
Write-Output "launched=$($p.Id) hwnd=$hwnd trigger=$trigger"

for ($i = 0; $i -lt 6; $i++) {
    Start-Sleep -Seconds 5
    if ($p.HasExited) { break }
    $hwnd = Move-NeXiumToBottom -Process $p
}

for ($i = 0; $i -lt 4; $i++) {
    if ($p.HasExited) { break }
    $hwnd = Move-NeXiumToBottom -Process $p
    if ($hwnd -ne [IntPtr]::Zero) {
        [void][NxCodexTraceWindow]::PostMessage(
            $hwnd,
            0x0100,
            [IntPtr]0x43,
            [IntPtr]([int64]0x002E0001)
        )
        Start-Sleep -Milliseconds 180
        [void][NxCodexTraceWindow]::PostMessage(
            $hwnd,
            0x0101,
            [IntPtr]0x43,
            [IntPtr]([int64]0xC02E0001)
        )
        Write-Output "controller-input-sent=$($i + 1)"
    }
    Start-Sleep -Seconds 8
}

for ($i = 0; $i -lt $LateLoops; $i++) {
    Start-Sleep -Seconds 5
    if ($p.HasExited) { break }
    $hwnd = Move-NeXiumToBottom -Process $p
}

if (-not $p.HasExited -and $hwnd -ne [IntPtr]::Zero) {
    $rect = [NxCodexTraceWindow+Rect]::new()
    if ([NxCodexTraceWindow]::GetWindowRect($hwnd, [ref]$rect)) {
        $width = $rect.Right - $rect.Left
        $height = $rect.Bottom - $rect.Top
        if ($width -gt 0 -and $height -gt 0) {
            $bitmap = [Drawing.Bitmap]::new($width, $height)
            $graphics = [Drawing.Graphics]::FromImage($bitmap)
            $hdc = $graphics.GetHdc()
            try {
                [void][NxCodexTraceWindow]::PrintWindow($hwnd, $hdc, 2)
            } finally {
                $graphics.ReleaseHdc($hdc)
                $graphics.Dispose()
            }
            $bitmap.Save($capture, [Drawing.Imaging.ImageFormat]::Png)
            $bitmap.Dispose()
            Write-Output "capture=$capture"
        }
    }
}

if (-not $p.HasExited) {
    $null = New-Item -ItemType File -Path $trigger -Force
    Write-Output 'draw-trigger-created'
}

for ($i = 0; $i -lt 3; $i++) {
    Start-Sleep -Seconds 5
    if ($p.HasExited) { break }
    $hwnd = Move-NeXiumToBottom -Process $p
}

if (-not $p.HasExited) {
    $null = $p.CloseMainWindow()
    if (-not $p.WaitForExit(8000)) {
        Stop-Process -Id $p.Id -Force
        $p.WaitForExit()
    }
}

Remove-Item -LiteralPath $trigger -Force -ErrorAction SilentlyContinue
$after = Get-ChildItem -LiteralPath $logs -Filter 'nexium-*.log' |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
$presentAfter = Get-ChildItem -LiteralPath $logs -Filter 'present-*.bmp' -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -ge $runStarted } |
    Sort-Object LastWriteTime
Write-Output "closed=$($p.Id) exit=$($p.ExitCode) log=$($after.FullName) previous=$($before.FullName)"
foreach ($frame in $presentAfter) {
    Write-Output "present=$($frame.FullName)"
}
