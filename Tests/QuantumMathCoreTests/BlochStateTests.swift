import Foundation
import Testing
@testable import QuantumMathCore

private struct EmptyResolver: ValueResolver {
    func resolve(_ reference: QuantumReferenceID) throws -> QuantumValue {
        throw QuantumMathError.operationNotDefined("Unexpected reference in Bloch state test: \(reference.rawValue)")
    }
}

struct BlochStateTests {
    private let config = QuantumMathConfig.ketStepsDefault

    @Test
    func densityFromBlochVectorBuildsValidTraceOneOperator() throws {
        let vector = QubitBlochVector(
            x: .zero,
            y: .zero,
            z: .zero
        )

        let density = try QuantumDomain.densityOperator(fromBlochVector: vector, config: config)
        let trace = try QuantumDomain.trace(density)
        let semantics = QuantumDomain.operatorSemantics(density, config: config)

        #expect(trace.approximatelyEquals(.one, epsilon: config.scalarComparisonEpsilon))
        if case let .density(summary) = semantics {
            #expect(summary.purity == .mixed)
            #expect(summary.qubitBlochVector?.x.approximatelyEquals(.zero, epsilon: config.scalarComparisonEpsilon) == true)
            #expect(summary.qubitBlochVector?.y.approximatelyEquals(.zero, epsilon: config.scalarComparisonEpsilon) == true)
            #expect(summary.qubitBlochVector?.z.approximatelyEquals(.zero, epsilon: config.scalarComparisonEpsilon) == true)
        } else {
            Issue.record("Expected a density semantic classification.")
        }
    }

    @Test
    func densityFromBlochVectorRejectsOutsideBall() throws {
        let vector = QubitBlochVector(
            x: Scalar.approx(ComplexNumber(re: 1.2, im: 0)),
            y: .zero,
            z: .zero
        )

        do {
            _ = try QuantumDomain.densityOperator(fromBlochVector: vector, config: config)
            Issue.record("Expected out-of-ball vector to fail.")
        } catch {
        }
    }

    @Test
    func ketFromBlochAnglesMatchesNorthPole() throws {
        let ket = try QuantumDomain.ket(
            fromBlochAngles: QubitBlochAngles(thetaRadians: .zero, phiRadians: .zero),
            config: config
        )

        #expect(ket.coefficients[0].approximatelyEquals(.one, epsilon: config.scalarComparisonEpsilon))
        #expect(ket.coefficients[1].approximatelyEquals(.zero, epsilon: config.scalarComparisonEpsilon))

        let vector = QuantumDomain.blochVector(for: ket, config: config)
        #expect(vector?.x.approximatelyEquals(.zero, epsilon: config.scalarComparisonEpsilon) == true)
        #expect(vector?.y.approximatelyEquals(.zero, epsilon: config.scalarComparisonEpsilon) == true)
        #expect(vector?.z.approximatelyEquals(.one, epsilon: config.scalarComparisonEpsilon) == true)
    }

    @Test
    func quantumExpressionEvaluatesBlochState() throws {
        let expression = QuantumExpression.blochState(
            .density(
                QubitBlochVector(
                    x: Scalar.approx(ComplexNumber(re: 0.5, im: 0)),
                    y: .zero,
                    z: .zero
                )
            )
        )
        let result = try QuantumOperation(config: config).evaluate(
            expression: expression,
            resolver: EmptyResolver()
        )

        guard case let .oper(operatorValue) = result else {
            Issue.record("Expected Bloch density expression to evaluate to an operator.")
            return
        }

        let trace = try QuantumDomain.trace(operatorValue)
        #expect(trace.approximatelyEquals(.one, epsilon: config.scalarComparisonEpsilon))
    }
}
