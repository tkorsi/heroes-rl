// swift-tools-version: 6.0
import PackageDescription
let package = Package(name: "HeroesReplay", platforms: [.macOS(.v14)], products: [.executable(name: "HeroesReplay", targets: ["HeroesReplay"])], targets: [.executableTarget(name: "HeroesReplay")])
