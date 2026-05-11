import Foundation
import Testing
@testable import QuantumMathCore

private struct MockHermitianSolver: HermitianEigenSolver {
    func decompose(_ matrix: Matrix<Scalar>) throws -> HermitianEigenDecompositionRaw {
        let dim = matrix.rows
        let eigenvalues = (0..<dim).map(Double.init)
        let eigenvectors = (0..<dim).flatMap { row in
            (0..<dim).map { col in
                ComplexNumber(re: row == col ? 1 : 0, im: 0)
            }
        }
        return HermitianEigenDecompositionRaw(
            dimension: dim,
            eigenvalues: eigenvalues,
            eigenvectors: eigenvectors
        )
    }
}

private struct MockSingularSolver: SingularValueSolver {
    func decompose(_ matrix: Matrix<Scalar>) throws -> SingularValueDecompositionRaw {
        let rank = min(matrix.rows, matrix.cols)
        let left = (0..<matrix.rows).flatMap { row in
            (0..<rank).map { col in
                ComplexNumber(re: row == col ? 1 : 0, im: 0)
            }
        }
        let right = (0..<matrix.cols).flatMap { row in
            (0..<rank).map { col in
                ComplexNumber(re: row == col ? 1 : 0, im: 0)
            }
        }
        return SingularValueDecompositionRaw(
            rows: matrix.rows,
            cols: matrix.cols,
            rank: rank,
            singularValues: (0..<rank).map { Double(rank - $0) },
            leftVectors: left,
            rightVectors: right
        )
    }
}

struct AnalyzerAndBoundaryTests {
    @Test
    func analyzersProduceExpectedShapesWithInjectedBackends() throws {
        let space = try Space(validating: [.qubit], maxComputableDimension: 16)
        let basis = Basis.computational(for: space)
        let identity = try Operator(
            domain: space,
            codomain: space,
            columnBasis: basis,
            rowBasis: basis,
            entries: Matrix.identity(size: 2)
        )

        let spectral = SpectralAnalyzer(
            config: .ketStepsDefault,
            backend: MockHermitianSolver()
        )
        let svd = SingularValueAnalyzer(
            config: .ketStepsDefault,
            backend: MockSingularSolver()
        )

        let spectralResult = try spectral.hermitianDecomposition(identity)
        let svdResult = try svd.decompose(identity)

        #expect(spectralResult.decomposition.components.count == 2)
        #expect(svdResult.decomposition.components.count == 2)
    }

    @Test
    func corePackageDoesNotImportForbiddenBackendModules() throws {
        let testFilePath = URL(fileURLWithPath: #filePath)
        let packageRoot = testFilePath
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()

        let sourceRoot = packageRoot.appendingPathComponent("Sources/QuantumMathCore")
        let fileURLs = try FileManager.default.contentsOfDirectory(
            at: sourceRoot,
            includingPropertiesForKeys: nil
        )
        let swiftFiles = fileURLs.filter { $0.pathExtension == "swift" }

        for file in swiftFiles {
            let text = try String(contentsOf: file, encoding: .utf8)
            #expect(!text.contains("import Accelerate"))
            #expect(!text.contains("KSHermitianEigenSolver"))
            #expect(!text.contains("KSSingularValueSolver"))
        }
    }
}
