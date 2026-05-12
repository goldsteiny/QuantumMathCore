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
        if let knownState = KnownStateExactifier.bestKnownState(matching: ket, config: config) {
            return StateVectorExactificationResult(
                ket: knownState,
                metadataByCoefficient: verifiedMetadata(count: knownState.coefficients.count)
            )
        }

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

private enum KnownStateExactifier {
    static func bestKnownState(matching ket: Ket, config: QuantumMathConfig) -> Ket? {
        let normSquared = QuantumDomain.normSquared(ket)
        let tolerance = max(config.projectorDistanceThreshold, config.spectralResidualThreshold)
        guard ket.basis.isComputationalCoordinateBasis,
              abs(normSquared - 1) <= tolerance,
              let candidates = try? candidates(for: ket.space, basis: ket.basis, config: config) else {
            return nil
        }

        return candidates
            .compactMap { candidate -> (ket: Ket, distance: Double)? in
                guard let distance = projectiveDistance(candidate, ket, config: config),
                      distance <= tolerance else {
                    return nil
                }
                return (candidate, distance)
            }
            .sorted { lhs, rhs in
                if lhs.distance != rhs.distance {
                    return lhs.distance < rhs.distance
                }
                return canonicalKey(for: lhs.ket) < canonicalKey(for: rhs.ket)
            }
            .first?
            .ket
    }

    private static func candidates(
        for space: Space,
        basis: Basis,
        config: QuantumMathConfig
    ) throws -> [Ket] {
        guard basis.isComputationalCoordinateBasis else {
            return []
        }

        var all = try atomicOrEntangledCandidates(for: space, basis: basis)
        if space.factors.count > 1 {
            all.append(contentsOf: try tensorProductCandidates(for: space, basis: basis, config: config))
        }

        var seen = Set<String>()
        return all.filter { seen.insert(canonicalKey(for: $0)).inserted }
    }

    private static func atomicOrEntangledCandidates(for space: Space, basis: Basis) throws -> [Ket] {
        if space.factors.count == 2,
           space.factors.allSatisfy({ $0.dimension == 2 }) {
            return try computationalBasisCandidates(for: space, basis: basis)
                + bellCandidates(space: space, basis: basis)
        }

        guard space.factors.count == 1, let dimension = space.factors.first?.dimension else {
            return try computationalBasisCandidates(for: space, basis: basis)
        }

        var candidates = try computationalBasisCandidates(for: space, basis: basis)
        switch dimension {
        case 2:
            candidates.append(contentsOf: try qubitAxisCandidates(space: space, basis: basis))
        case 3:
            candidates.append(contentsOf: try qutritFourierCandidates(space: space, basis: basis))
        case 4:
            candidates.append(contentsOf: try c4FourierCandidates(space: space, basis: basis))
        default:
            break
        }
        return candidates
    }

    private static func computationalBasisCandidates(for space: Space, basis: Basis) throws -> [Ket] {
        try (0..<space.dimension).map { index in
            var coefficients = Array(repeating: Scalar.zero, count: space.dimension)
            coefficients[index] = .one
            return try Ket(space: space, basis: basis, coefficients: coefficients)
        }
    }

    private static func qubitAxisCandidates(space: Space, basis: Basis) throws -> [Ket] {
        let halfRootTwo = realRadical(1, 2, 2)
        let plus = scalar(halfRootTwo)
        let minus = scalar(realRadical(-1, 2, 2))
        let plusI = scalar(real: .rational(.zero), imag: halfRootTwo)
        let minusI = scalar(real: .rational(.zero), imag: realRadical(-1, 2, 2))

        return try [
            [plus, plus],
            [plus, minus],
            [plus, plusI],
            [plus, minusI]
        ].map { try Ket(space: space, basis: basis, coefficients: $0) }
    }

    private static func qutritFourierCandidates(space: Space, basis: Basis) throws -> [Ket] {
        let invSqrt3 = scalar(realRadical(1, 3, 3))
        let omegaOverSqrt3 = scalar(
            real: realRadical(-1, 6, 3),
            imag: .rational(Rational(1, 2))
        )
        let omegaSquaredOverSqrt3 = scalar(
            real: realRadical(-1, 6, 3),
            imag: .rational(Rational(-1, 2))
        )

        return try [
            [invSqrt3, invSqrt3, invSqrt3],
            [invSqrt3, omegaOverSqrt3, omegaSquaredOverSqrt3],
            [invSqrt3, omegaSquaredOverSqrt3, omegaOverSqrt3]
        ].map { try Ket(space: space, basis: basis, coefficients: $0) }
    }

    private static func c4FourierCandidates(space: Space, basis: Basis) throws -> [Ket] {
        let half = Scalar.exact(.rational(Rational(1, 2)))
        let minusHalf = Scalar.exact(.rational(Rational(-1, 2)))
        let halfI = scalar(real: .rational(.zero), imag: .rational(Rational(1, 2)))
        let minusHalfI = scalar(real: .rational(.zero), imag: .rational(Rational(-1, 2)))

        return try [
            [half, half, half, half],
            [half, halfI, minusHalf, minusHalfI],
            [half, minusHalf, half, minusHalf],
            [half, minusHalfI, minusHalf, halfI]
        ].map { try Ket(space: space, basis: basis, coefficients: $0) }
    }

    private static func bellCandidates(space: Space, basis: Basis) throws -> [Ket] {
        let invSqrt2 = scalar(realRadical(1, 2, 2))
        let minusInvSqrt2 = scalar(realRadical(-1, 2, 2))
        let iInvSqrt2 = scalar(real: .rational(.zero), imag: realRadical(1, 2, 2))
        let minusIInvSqrt2 = scalar(real: .rational(.zero), imag: realRadical(-1, 2, 2))

        let zero = Scalar.zero
        return try [
            [invSqrt2, zero, zero, invSqrt2],
            [invSqrt2, zero, zero, minusInvSqrt2],
            [zero, invSqrt2, invSqrt2, zero],
            [zero, invSqrt2, minusInvSqrt2, zero],
            [invSqrt2, zero, zero, iInvSqrt2],
            [invSqrt2, zero, zero, minusIInvSqrt2],
            [zero, invSqrt2, iInvSqrt2, zero],
            [zero, invSqrt2, minusIInvSqrt2, zero]
        ].map { try Ket(space: space, basis: basis, coefficients: $0) }
    }

    private static func tensorProductCandidates(
        for space: Space,
        basis: Basis,
        config: QuantumMathConfig
    ) throws -> [Ket] {
        var partials: [Ket] = []

        for factor in space.factors {
            let factorSpace = Space.atomic(factor)
            let factorBasis = Basis.computational(for: factorSpace)
            let factorCandidates = try atomicOrEntangledCandidates(for: factorSpace, basis: factorBasis)

            if partials.isEmpty {
                partials = factorCandidates
            } else {
                let nextCount = partials.count * factorCandidates.count
                guard nextCount <= 4_096 else {
                    return []
                }
                partials = try partials.flatMap { partial in
                    try factorCandidates.map { try tensor(partial, $0, config: config) }
                }
            }
        }

        return try partials.map { candidate in
            try Ket(space: space, basis: basis, coefficients: candidate.coefficients)
        }
    }

    private static func tensor(_ lhs: Ket, _ rhs: Ket, config: QuantumMathConfig) throws -> Ket {
        let space = try Space(validating: lhs.space.factors + rhs.space.factors, maxComputableDimension: config.maxComputableDimension)
        let basis = try Basis(space: space, factorKinds: lhs.basis.factorKinds + rhs.basis.factorKinds)
        let coefficients = lhs.coefficients.flatMap { lhsScalar in
            rhs.coefficients.map { rhsScalar in
                ScalarExactificationAdapter.exactify(
                    lhsScalar * rhsScalar,
                    label: "known.tensor",
                    config: config
                ).scalar
            }
        }
        return try Ket(space: space, basis: basis, coefficients: coefficients)
    }

    private static func projectiveDistance(_ lhs: Ket, _ rhs: Ket, config: QuantumMathConfig) -> Double? {
        guard lhs.space.isCoordinateCompatible(with: rhs.space),
              lhs.basis.isCoordinateCompatible(with: rhs.basis),
              lhs.coefficients.count == rhs.coefficients.count else {
            return nil
        }

        let lhsNorm = lhs.coefficients.reduce(0.0) { $0 + $1.magnitudeSquaredApproximate }
        let rhsNorm = rhs.coefficients.reduce(0.0) { $0 + $1.magnitudeSquaredApproximate }
        guard lhsNorm > config.scalarComparisonEpsilon,
              rhsNorm > config.scalarComparisonEpsilon else {
            return nil
        }

        let overlap = zip(lhs.coefficients, rhs.coefficients).reduce(
            ComplexNumber(re: 0, im: 0)
        ) { partial, pair in
            partial + (pair.0.approximateValue.conjugated * pair.1.approximateValue)
        }
        let fidelity = min(1, overlap.magnitudeSquared / (lhsNorm * rhsNorm))
        return abs(1 - fidelity)
    }

    private static func realRadical(_ numerator: Int, _ denominator: Int, _ radicand: Int) -> ExactRealExpression {
        .canonicalRationalTimesSqrt(coefficient: Rational(numerator, denominator), radicand: radicand)
    }

    private static func scalar(_ real: ExactRealExpression) -> Scalar {
        switch real.canonicalized {
        case let .rational(value):
            return .exact(.rational(value))
        case let .rationalTimesSqrt(coefficient, radicand):
            return .exact(.signedRationalTimesSqrt(coefficient: coefficient, radicand: radicand))
        }
    }

    private static func scalar(real: ExactRealExpression, imag: ExactRealExpression) -> Scalar {
        .exact(.complex(real: real, imag: imag).canonicalized)
    }

    private static func canonicalKey(for ket: Ket) -> String {
        ket.coefficients.map { scalar in
            switch scalar {
            case let .exact(expression):
                return expression.canonicalKey
            case let .approx(value):
                return "\(value.re),\(value.im)"
            }
        }.joined(separator: "|")
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

private func verifiedMetadata(count: Int) -> [Int: ExactificationMetadata] {
    Dictionary(
        uniqueKeysWithValues: (0..<count).map {
            ($0, ExactificationMetadata(status: .verifiedExact, witness: nil))
        }
    )
}
