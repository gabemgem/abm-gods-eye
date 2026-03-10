enablePlugins(org.nlogo.build.NetLogoExtension)

name := "gods-eye"
version := "0.1.0"
scalaVersion := "2.12.19"

netLogoVersion      := "6.4.0"
netLogoClassManager := "GodsEyeExtension"
netLogoExtName      := "gods-eye"

// gson for JSON serialisation — bundled into the extension folder
libraryDependencies += "com.google.code.gson" % "gson" % "2.11.0"

// Place the built jar and dependencies in the project root so sbt package
// produces a distributable zip.
enablePlugins(org.nlogo.build.NetLogoExtension)
