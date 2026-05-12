import Foundation
import ExactValueRecoveryCore
import Testing
@testable import QuantumMathCore

struct WeylUnitaryErrorBasisTests {
    @Test
    func qubitWeylBasisMatchesPauliConvention() throws {
        let basis = try QuantumDomain.weylUnitaryErrorBasis(dimension: 2)

        let identity = try basis.element(for: WeylIndex(dimension: 2, shift: 0, phase: 0)).operatorValue
        let x = try basis.element(for: WeylIndex(dimension: 2, shift: 1, phase: 0)).operatorValue
        let z = try basis.element(for: WeylIndex(dimension: 2, shift: 0, phase: 1)).operatorValue
        let xz = try basis.element(for: WeylIndex(dimension: 2, shift: 1, phase: 1)).operatorValue

        #expect(identity.entries.approximatelyEquals(.identity(size: 2), epsilon: 1e-10))
        #expect(x.entries[0, 1].approximatelyEquals(.one, epsilon: 1e-10))
        #expect(x.entries[1, 0].approximatelyEquals(.one, epsilon: 1e-10))
        #expect(z.entries[0, 0].approximatelyEquals(.one, epsilon: 1e-10))
        #expect(z.entries[1, 1].approximatelyEquals(Scalar(real: Rational(-1)), epsilon: 1e-10))
        #expect(xz.entries[1, 0].approximatelyEquals(.one, epsilon: 1e-10))
        #expect(xz.entries[0, 1].approximatelyEquals(Scalar(real: Rational(-1)), epsilon: 1e-10))
    }

    @Test
    func qutritWeylBasisIsHilbertSchmidtOrthonormal() throws {
        let basis = try QuantumDomain.weylUnitaryErrorBasis(dimension: 3, config: .ketStepsQutrit27Preview)

        #expect(basis.elements.count == 9)
        try basis.validateHilbertSchmidtOrthogonality(config: .ketStepsQutrit27Preview)
        for lhs in basis.elements {
            #expect(lhs.operatorValue.entries.isUnitary(tolerance: 1e-9))
            for rhs in basis.elements {
                let inner = try QuantumDomain.hilbertSchmidtInnerProduct(
                    lhs.operatorValue,
                    rhs.operatorValue
                )
                let expected = lhs.index == rhs.index ? Scalar(real: Rational(3)) : .zero
                #expect(inner.approximatelyEquals(expected, epsilon: 1e-9))
            }
        }
    }

    @Test
    func higherDimensionMatrixInspectionCreatesFullBasis() throws {
        let basis = try QuantumDomain.weylUnitaryErrorBasis(dimension: 5, config: .ketStepsQutrit27Preview)

        #expect(basis.elements.count == 25)
        #expect(basis.elements.first?.index.flatIndex == 0)
        #expect(basis.elements.last?.index.flatIndex == 24)
        #expect(try basis.element(flatIndex: 17).index == WeylIndex(dimension: 5, flatIndex: 17))
    }

    @Test
    func denseCodingDecodesEveryQubitAndQutritWeylIndex() throws {
        for dimension in [2, 3] {
            for phase in 0..<dimension {
                for shift in 0..<dimension {
                    let index = try WeylIndex(dimension: dimension, shift: shift, phase: phase)
                    let scheme = try QuantumDomain.denseCodingScheme(
                        dimension: dimension,
                        encodedIndex: index,
                        config: .ketStepsQutrit27Preview
                    )

                    #expect(scheme.decodedIndex == index)
                    #expect(scheme.factorOrder == [.aliceShare, .bobShare])
                }
            }
        }
    }

    @Test
    func teleportationCorrectionsRecoverQubitAndQutritBasisStates() throws {
        for dimension in [2, 3] {
            let basis = try oneQuditBasis(dimension: dimension)
            for index in 0..<dimension {
                var coefficients = Array(repeating: Scalar.zero, count: dimension)
                coefficients[index] = .one
                let input = try Ket(space: basis.space, basis: basis, coefficients: coefficients)

                let scheme = try QuantumDomain.teleportationScheme(
                    inputState: input,
                    config: .ketStepsQutrit27Preview
                )

                #expect(scheme.outcomes.count == dimension * dimension)
                #expect(scheme.factorOrder == [.input, .aliceShare, .bobShare])
                for outcome in scheme.outcomes {
                    #expect(outcome.probability.approximatelyEquals(
                        Scalar(real: Rational(1, dimension * dimension)),
                        epsilon: 1e-10
                    ))
                    #expect(sameProjectiveRay(outcome.bobStateAfterCorrection, input))
                }
            }
        }
    }

    @Test
    func teleportationCorrectionRecoversQutritSuperposition() throws {
        let basis = try oneQuditBasis(dimension: 3)
        let inverseRootThree = try QuantumDomain.inverseSquareRoot(3)
        let input = try Ket(
            space: basis.space,
            basis: basis,
            coefficients: [inverseRootThree, inverseRootThree, inverseRootThree]
        )

        let scheme = try QuantumDomain.teleportationScheme(
            inputState: input,
            config: .ketStepsQutrit27Preview
        )

        for outcome in scheme.outcomes {
            #expect(sameProjectiveRay(outcome.bobStateAfterCorrection, input))
        }
    }

    @Test
    func dimensionFourTeleportationExceedsQutritPreviewLimit() throws {
        let basis = try oneQuditBasis(dimension: 4)
        let input = try Ket(
            space: basis.space,
            basis: basis,
            coefficients: [.one, .zero, .zero, .zero]
        )

        do {
            _ = try QuantumDomain.teleportationScheme(
                inputState: input,
                config: .ketStepsQutrit27Preview
            )
            Issue.record("Dimension-4 teleportation should exceed max total dimension 27.")
        } catch {
        }
    }

    private func oneQuditBasis(dimension: Int) throws -> Basis {
        let space = try Space(
            validating: [try AtomicSpace(dimension: dimension)],
            maxComputableDimension: QuantumMathConfig.ketStepsQutrit27Preview.maxComputableDimension
        )
        return .computational(for: space)
    }

    private func sameProjectiveRay(_ lhs: Ket, _ rhs: Ket) -> Bool {
        guard lhs.space == rhs.space,
              lhs.basis == rhs.basis,
              lhs.coefficients.count == rhs.coefficients.count else {
            return false
        }

        let epsilon = 1e-8
        var factor: Scalar?
        for pair in zip(lhs.coefficients, rhs.coefficients) {
            switch (pair.0.isZero(epsilon: epsilon), pair.1.isZero(epsilon: epsilon)) {
            case (true, true):
                continue
            case (true, false), (false, true):
                return false
            case (false, false):
                factor = pair.0.divided(by: pair.1)
            }
            if factor != nil {
                break
            }
        }

        guard let factor else {
            return false
        }
        return zip(lhs.coefficients, rhs.coefficients).allSatisfy { pair in
            pair.0.approximatelyEquals(factor * pair.1, epsilon: epsilon)
        }
    }
}
