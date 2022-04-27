$Target="muon-airss-out-uep/uep"

$SubFolders = Get-ChildItem -Path $Target -Directory

Foreach($SubFolder in $SubFolders)
{
    $BaseName = Get-Item $SubFolder.FullName | % basename
    pm-uep-opt $Target/$SubFolder/$BaseName.yaml
}
