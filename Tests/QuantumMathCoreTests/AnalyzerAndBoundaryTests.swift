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

private struct DictionaryResolver: ValueResolver {
    let values: [QuantumReferenceID: QuantumValue]

    func resolve(_ reference: QuantumReferenceID) throws -> QuantumValue {
        guard let value = values[reference] else {
            throw QuantumMathError.operationNotDefined("Missing test reference.")
        }
        return value
    }
}

struct AnalyzerAndBoundaryTests {
    @Test
    func quantumOperationEvaluatesTypedExpressionThroughResolver() throws {
        let space = try Space(validating: [.qubit], maxComputableDimension: 16)
        let basis = Basis.computational(for: space)
        let ket = try Ket(space: space, basis: basis, coefficients: [.one, .zero])
        let identity = try Operator(
            domain: space,
            codomain: space,
            columnBasis: basis,
            rowBasis: basis,
            entries: .identity(size: 2)
        )
        let ketID = QuantumReferenceID("ket")
        let operatorID = QuantumReferenceID("identity")
        let resolver = DictionaryResolver(values: [
            ketID: .ket(ket),
            operatorID: .oper(identity)
        ])

        let result = try QuantumOperation().evaluate(
            expression: .binary(.applyOperator, operatorID, ketID),
            resolver: resolver
        )

        guard case let .ket(resultKet) = result else {
            Issue.record("Expected ket result.")
            return
        }

        #expect(resultKet.coefficients[0].approximatelyEquals(.one, epsilon: 1e-10))
        #expect(resultKet.coefficients[1].approximatelyEquals(.zero, epsilon: 1e-10))
    }

    @Test
    func qutrit27PreviewConfigAcceptsThreeQutritSpace() throws {
        let qutritSpace = try Space(
            validating: [.qutrit, .qutrit, .qutrit],
            maxComputableDimension: QuantumMathConfig.ketStepsQutrit27Preview.maxComputableDimension
        )

        #expect(qutritSpace.dimension == 27)
    }

    @Test
    func qutrit27PreviewIsRequiredForThreeQutritTensorEvaluation() throws {
        let qutrit = try Space(validating: [.qutrit], maxComputableDimension: 16)
        let basis = Basis.computational(for: qutrit)
        let ket = try Ket(space: qutrit, basis: basis, coefficients: [.one, .zero, .zero])
        let ids = [
            QuantumReferenceID("q0"),
            QuantumReferenceID("q1"),
            QuantumReferenceID("q2")
        ]
        let resolver = DictionaryResolver(values: Dictionary(
            uniqueKeysWithValues: ids.map { ($0, QuantumValue.ket(ket)) }
        ))
        let expression = QuantumExpression.ordered(.tensor, ids)

        do {
            _ = try QuantumOperation().evaluate(expression: expression, resolver: resolver)
            Issue.record("Default KetSteps config should reject three qutrit tensor evaluation.")
        } catch {
        }

        let result = try QuantumOperation(config: .ketStepsQutrit27Preview).evaluate(
            expression: expression,
            resolver: resolver
        )

        guard case let .ket(resultKet) = result else {
            Issue.record("Expected qutrit preview tensor to evaluate to a ket.")
            return
        }

        #expect(resultKet.space.dimension == 27)
    }

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
        #expect(spectralResult.decomposition.components.allSatisfy { $0.eigenvalue.isApproximate })
        #expect(spectralResult.decomposition.components.allSatisfy { component in
            component.eigenvectors.contains { vector in
                vector.coefficients.contains(where: \.isApproximate)
            }
        })
        #expect(svdResult.decomposition.components.allSatisfy { $0.singularValue.isApproximate })
        #expect(svdResult.decomposition.components.allSatisfy { component in
            component.leftVector.coefficients.contains(where: \.isApproximate)
                || component.rightVector.coefficients.contains(where: \.isApproximate)
        })
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
