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

    private func angleScalar(_ value: Double) -> Scalar {
        .approx(ComplexNumber(re: value, im: 0))
    }

    @Test
    func plusIKetLandsOnPositiveYAxis() throws {
        // Regression: the ket→Bloch path used α·β̄ and flipped the sign of y.
        // |+i⟩ = (|0⟩ + i|1⟩)/√2 must land on +y under the standard convention.
        let ket = try QuantumDomain.ket(
            fromBlochAngles: QubitBlochAngles(
                thetaRadians: angleScalar(.pi / 2),
                phiRadians: angleScalar(.pi / 2)
            ),
            config: config
        )
        let vector = try #require(QuantumDomain.blochVector(for: ket, config: config))
        #expect(vector.x.approximatelyEquals(.zero, epsilon: config.scalarComparisonEpsilon))
        #expect(vector.y.approximatelyEquals(.one, epsilon: config.scalarComparisonEpsilon))
        #expect(vector.z.approximatelyEquals(.zero, epsilon: config.scalarComparisonEpsilon))
    }

    @Test
    func ketAndDensityBlochPathsAgreeWithSphericalClosedForm() throws {
        // Both conversion paths are anchored to the same ground truth
        // (x, y, z) = r·(sinθ·cosφ, sinθ·sinφ, cosθ), so a sign/convention
        // divergence in EITHER path fails this test.
        let theta = 1.1
        let phi = 0.7
        let expected = (x: sin(theta) * cos(phi), y: sin(theta) * sin(phi), z: cos(theta))
        let epsilon = config.scalarComparisonEpsilon

        // Ket path (r = 1): constructor convention vs ket→Bloch conversion.
        let ket = try QuantumDomain.ket(
            fromBlochAngles: QubitBlochAngles(
                thetaRadians: angleScalar(theta),
                phiRadians: angleScalar(phi)
            ),
            config: config
        )
        let viaKet = try #require(QuantumDomain.blochVector(for: ket, config: config))
        #expect(viaKet.x.approximatelyEquals(angleScalar(expected.x), epsilon: epsilon))
        #expect(viaKet.y.approximatelyEquals(angleScalar(expected.y), epsilon: epsilon))
        #expect(viaKet.z.approximatelyEquals(angleScalar(expected.z), epsilon: epsilon))

        // Density path: vector → ρ → vector must round-trip through the
        // density-operator conversion. The vector must stay inside the region
        // the (intentionally conservative) Gershgorin PSD certification accepts:
        // min((1±z)/2) ≥ √(x²+y²)/2 — an equatorial r = 0.6 vector qualifies
        // and still exercises the signs of both x and y.
        let radius = 0.6
        let input = QubitBlochVector(
            x: angleScalar(radius * cos(phi)),
            y: angleScalar(radius * sin(phi)),
            z: angleScalar(0)
        )
        let density = try QuantumDomain.densityOperator(fromBlochVector: input, config: config)
        guard case let .density(summary) = QuantumDomain.operatorSemantics(density, config: config),
              let viaDensity = summary.qubitBlochVector else {
            Issue.record("Expected the Bloch-ball density operator to classify with a Bloch vector.")
            return
        }
        #expect(viaDensity.x.approximatelyEquals(input.x, epsilon: epsilon))
        #expect(viaDensity.y.approximatelyEquals(input.y, epsilon: epsilon))
        #expect(viaDensity.z.approximatelyEquals(input.z, epsilon: epsilon))
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
