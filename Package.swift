// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "QuantumMathCore",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "QuantumMathCore",
            targets: ["QuantumMathCore"]
        )
    ],
    dependencies: [
        .package(name: "exactvaluerecovery", path: "../ExactValueRecovery")
    ],
    targets: [
        .target(
            name: "QuantumMathCore",
            dependencies: [
                .product(name: "ExactValueRecoveryCore", package: "exactvaluerecovery")
            ]
        ),
        .testTarget(
            name: "QuantumMathCoreTests",
            dependencies: ["QuantumMathCore"]
        )
    ]
)
