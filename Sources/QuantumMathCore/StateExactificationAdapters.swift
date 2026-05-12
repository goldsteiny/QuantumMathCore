import Foundation
import ExactValueRecoveryCore

public struct StateVectorExactificationResult: Hashable, Sendable, Codable {
    public let ket: Ket
    public let metadataByCoefficient: [Int: ExactificationMetadata]

    public init(
        ket: Ket,
        metadataByCoefficient: [Int: ExactificationMetadata]
    ) {
        self.ket = ket
        self.metadataByCoefficient = metadataByCoefficient
    }
}

public enum StateVectorExactificationAdapter {
    public static func exactify(
        _ ket: Ket,
        label: String,
        config: QuantumMathConfig
    ) -> StateVectorExactificationResult {
        var metadata: [Int: ExactificationMetadata] = [:]
        let coefficients = ket.coefficients.enumerated().map { index, coefficient in
            let exactified = ScalarExactificationAdapter.exactify(
                coefficient,
                label: "\(label).c\(index)",
                config: config
            )
            metadata[index] = metadataFor(outcome: exactified.outcome)
            return exactified.scalar
        }

        guard let exactifiedKet = try? Ket(
            space: ket.space,
            basis: ket.basis,
            coefficients: coefficients
        ) else {
            return StateVectorExactificationResult(
                ket: ket,
                metadataByCoefficient: metadata
            )
        }

        return StateVectorExactificationResult(
            ket: exactifiedKet,
            metadataByCoefficient: metadata
        )
    }
}

private func metadataFor(
    outcome: ExactificationOutcome<QuantumVerificationWitness>
) -> ExactificationMetadata {
    switch outcome {
    case let .exact(_, witness):
        return ExactificationMetadata(status: .verifiedExact, witness: witness)
    case let .mixed(_, witness, _):
        return ExactificationMetadata(status: .verifiedExact, witness: witness)
    case let .unresolved(reason):
        return ExactificationMetadata(status: .unresolved(reason), witness: nil)
    }
}
