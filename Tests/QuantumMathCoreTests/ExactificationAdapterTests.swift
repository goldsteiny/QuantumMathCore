import Foundation
import Testing
@testable import QuantumMathCore

struct ExactificationAdapterTests {
    @Test
    func scalarExactificationRecoversSimpleRational() {
        let result = ScalarExactificationAdapter.exactify(
            .approx(ComplexNumber(re: 0.5, im: 0)),
            label: "unit-test",
            config: .ketStepsDefault
        )

        #expect(!result.scalar.isApproximate)
    }

    @Test
    func stateVectorExactificationRecoversSimpleRadicalCoefficients() throws {
        let space = try Space(validating: [.qubit], maxComputableDimension: 16)
        let basis = Basis.computational(for: space)
        let inverseRootTwo = 1 / Foundation.sqrt(2)
        let approximatePlus = try Ket(
            space: space,
            basis: basis,
            coefficients: [
                .approx(ComplexNumber(re: inverseRootTwo, im: 0)),
                .approx(ComplexNumber(re: inverseRootTwo, im: 0))
            ]
        )

        let result = StateVectorExactificationAdapter.exactify(
            approximatePlus,
            label: "plus",
            config: .ketStepsDefault
        )

        #expect(result.ket.coefficients.allSatisfy { !$0.isApproximate })
    }
}
