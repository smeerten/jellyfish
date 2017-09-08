Set oWS = WScript.CreateObject("WScript.Shell") 
set fso = CreateObject("Scripting.FileSystemObject")
strDesktop= oWS.SpecialFolders("Desktop") 
appDir = fso.GetAbsolutePathName(".")
sLinkFile = strDesktop + "\Jellyfish.lnk"  
Set oLink = oWS.CreateShortcut(sLinkFile) 
Target = appDir + "\WindowsRun.bat"
oLink.TargetPath = """"& Target &"""" 
oLink.WorkingDirectory = appDir
oLink.Save 

' StartMenu
strStartMenu= oWS.SpecialFolders("Programs")
StartLocation = strStartMenu+"\Jellyfish.lnk"
 
Set oLink = oWS.CreateShortcut(StartLocation) 
oLink.TargetPath = """"& Target &"""" 
oLink.WorkingDirectory = appDir
oLink.Save 
