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
}
