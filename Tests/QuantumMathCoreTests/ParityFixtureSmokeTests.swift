import Foundation
import Testing
@testable import QuantumMathCore

private struct DegenerateHermitianSolver: HermitianEigenSolver {
    func decompose(_ matrix: Matrix<Scalar>) throws -> HermitianEigenDecompositionRaw {
        HermitianEigenDecompositionRaw(
            dimension: 2,
            eigenvalues: [1, 1],
            eigenvectors: [
                ComplexNumber(re: 1, im: 0), ComplexNumber(re: 0, im: 0),
                ComplexNumber(re: 0, im: 0), ComplexNumber(re: 1, im: 0)
            ]
        )
    }
}

private struct RationalHermitianSolver: HermitianEigenSolver {
    func decompose(_ matrix: Matrix<Scalar>) throws -> HermitianEigenDecompositionRaw {
        HermitianEigenDecompositionRaw(
            dimension: 2,
            eigenvalues: [0.5, 1.5],
            eigenvectors: [
                ComplexNumber(re: 1, im: 0), ComplexNumber(re: 0, im: 0),
                ComplexNumber(re: 0, im: 0), ComplexNumber(re: 1, im: 0)
            ]
        )
    }
}

struct ParityFixtureSmokeTests {
    @Test
    func spectralAnalyzerGroupsDegenerateSubspaceIntoSingleComponent() throws {
        let space = try Space(validating: [.qubit], maxComputableDimension: 16)
        let basis = Basis.computational(for: space)
        let identity = try Operator(
            domain: space,
            codomain: space,
            columnBasis: basis,
            rowBasis: basis,
            entries: Matrix.identity(size: 2)
        )

        let analyzer = SpectralAnalyzer(
            config: .ketStepsDefault,
            backend: DegenerateHermitianSolver()
        )

        let result = try analyzer.hermitianDecomposition(identity)
        #expect(result.decomposition.components.count == 1)

        let component = result.decomposition.components[0]
        #expect(component.multiplicity == 2)
        #expect(component.eigenvectors.count == 2)
    }

    @Test
    func spectralAnalyzerExactifiesSimpleRationalEigenvalues() throws {
        let space = try Space(validating: [.qubit], maxComputableDimension: 16)
        let basis = Basis.computational(for: space)
        let identity = try Operator(
            domain: space,
            codomain: space,
            columnBasis: basis,
            rowBasis: basis,
            entries: Matrix.identity(size: 2)
        )

        let analyzer = SpectralAnalyzer(
            config: .ketStepsDefault,
            backend: RationalHermitianSolver()
        )

        let result = try analyzer.hermitianDecomposition(identity)
        #expect(result.decomposition.components.count == 2)
        #expect(result.decomposition.components.allSatisfy { !$0.eigenvalue.isApproximate })
    }
}
