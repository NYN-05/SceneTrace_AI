$ports = @(8000, 5174, 5173, 8001)
foreach ($port in $ports) {
    $connections = netstat -ano | Select-String ":$port\s"
    if ($connections) {
        $killed = @{}
        foreach ($conn in $connections) {
            $parts = ($conn -split '\s+') | Where-Object { $_ -ne '' }
            $procId = $parts[-1]
            if ($procId -gt 0 -and !$killed.ContainsKey($procId)) {
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                $killed[$procId] = $true
                Write-Output "Killed PID $procId on port $port"
            }
        }
        if ($killed.Count -eq 0) { Write-Output "No process on port $port" }
    } else {
        Write-Output "No process on port $port"
    }
}
