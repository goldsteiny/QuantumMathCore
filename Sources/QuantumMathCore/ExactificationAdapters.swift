import Foundation
import ExactValueRecoveryCore

public struct QuantumVerificationWitness: Hashable, Sendable, Codable {
    public let evaluatedConstraints: [VerificationConstraintTag]
    public let maxResidual: Double
    public let maxThreshold: Double
    public let exactifiedVariables: [RecoveryVariableID]
    public let approximateVariables: [RecoveryVariableID]
    public let selectedCanonicalKeys: [CandidateCanonicalKey]
    public let verifierID: String

    public init(
        evaluatedConstraints: [VerificationConstraintTag],
        maxResidual: Double,
        maxThreshold: Double,
        exactifiedVariables: [RecoveryVariableID],
        approximateVariables: [RecoveryVariableID],
        selectedCanonicalKeys: [CandidateCanonicalKey],
        verifierID: String
    ) {
        self.evaluatedConstraints = evaluatedConstraints
        self.maxResidual = maxResidual
        self.maxThreshold = maxThreshold
        self.exactifiedVariables = exactifiedVariables.sorted()
        self.approximateVariables = approximateVariables.sorted()
        self.selectedCanonicalKeys = selectedCanonicalKeys.sorted()
        self.verifierID = verifierID
    }
}

public struct ScalarExactificationResult: Sendable, Codable {
    public let scalar: Scalar
    public let outcome: ExactificationOutcome<QuantumVerificationWitness>

    public init(
        scalar: Scalar,
        outcome: ExactificationOutcome<QuantumVerificationWitness>
    ) {
        self.scalar = scalar
        self.outcome = outcome
    }
}

public enum ScalarExactificationAdapter {
    public static func exactify(
        _ scalar: Scalar,
        label: String,
        config: QuantumMathConfig
    ) -> ScalarExactificationResult {
        let atomID = ApproximateAtomID("scalar/\(label)")
        let variableID = RecoveryVariableID("var/\(label)")
        let approximate = scalar.approximateValue

        guard approximate.re.isFinite, approximate.im.isFinite else {
            return ScalarExactificationResult(
                scalar: scalar,
                outcome: .unresolved(.nonFiniteInput([atomID]))
            )
        }

        let atom = ApproximateAtom(
            id: atomID,
            value: approximate.canonicalizedSignedZero,
            role: .scalar
        )
        let variable = RecoveryVariable(
            id: variableID,
            atomIDs: [atomID],
            priority: .normal
        )

        let provider = AnyCandidateProvider<ScalarVerificationContext>(
            providerID: "quantum.scalar.default"
        ) { variable, atoms, context, _ in
            guard let atom = atoms[atomID] else {
                return []
            }
            return candidates(
                for: atom,
                variableID: variable.id,
                providerID: "quantum.scalar.default",
                threshold: context.threshold
            )
        }

        let verifier = AnyRecoveryVerifier<ScalarVerificationContext, QuantumVerificationWitness> {
            assignment, variables, atoms, context in
            let evaluated = VerificationConstraintTag("scalar-distance")
            guard let binding = assignment.bindings.first,
                  let replacement = binding.atomReplacements.first,
                  let atom = atoms[replacement.atomID] else {
                return .rejected(
                    VerificationRejection(
                        failedConstraints: [
                            VerificationConstraintFailure(
                                tag: evaluated,
                                residual: nil,
                                threshold: context.threshold
                            )
                        ],
                        maxResidual: .infinity
                    )
                )
            }

            let recovered = replacement.expression.approximateValue
            let original = atom.value
            let residual = max(abs(recovered.re - original.re), abs(recovered.im - original.im))
            if residual > context.threshold {
                return .rejected(
                    VerificationRejection(
                        failedConstraints: [
                            VerificationConstraintFailure(
                                tag: evaluated,
                                residual: residual,
                                threshold: context.threshold
                            )
                        ],
                        maxResidual: residual
                    )
                )
            }

            let witness = QuantumVerificationWitness(
                evaluatedConstraints: [evaluated],
                maxResidual: residual,
                maxThreshold: context.threshold,
                exactifiedVariables: assignment.bindings.map(\.variableID),
                approximateVariables: assignment.unresolved(comparedTo: variables),
                selectedCanonicalKeys: assignment.bindings.map(\.canonicalKey),
                verifierID: context.verifierID
            )
            return .accepted(witness)
        }

        let context = ScalarVerificationContext(
            threshold: config.spectralResidualThreshold,
            verifierID: "quantum.scalar.v1"
        )

        let outcome = ExactificationEngine.solve(
            problem: ExactificationProblem(
                variables: [variable],
                atoms: [atom],
                candidateProviders: [provider],
                policy: config.exactificationPolicy,
                verifierContext: context
            ),
            verifier: verifier
        )

        switch outcome {
        case let .exact(assignment, _), let .mixed(assignment, _, _):
            if let replacement = assignment.bindings.first?.atomReplacements.first {
                return ScalarExactificationResult(
                    scalar: .exact(replacement.expression),
                    outcome: outcome
                )
            }
            return ScalarExactificationResult(scalar: scalar, outcome: outcome)
        case .unresolved:
            return ScalarExactificationResult(scalar: scalar, outcome: outcome)
        }
    }

    private static func candidates(
        for atom: ApproximateAtom,
        variableID: RecoveryVariableID,
        providerID: String,
        threshold: Double
    ) -> [CandidateBinding] {
        let target = atom.value
        var expressions: [ExactScalarExpression] = []
        expressions.append(.rational(.zero))
        expressions.append(.rational(.one))

        if abs(target.im) <= threshold {
            expressions.append(contentsOf: realExpressions(near: target.re).map { scalarExpression(for: $0) })
        } else {
            let realCandidates = realExpressions(near: target.re)
            let imaginaryCandidates = realExpressions(near: target.im)
            for real in realCandidates {
                for imaginary in imaginaryCandidates {
                    expressions.append(.complex(real: real, imag: imaginary))
                }
            }
        }

        let magnitude = Foundation.sqrt(target.magnitudeSquared)
        if abs(magnitude - 1) <= threshold * 4 {
            for order in [2, 3, 4, 6, 8, 12] {
                for power in 0..<order {
                    expressions.append(.rootOfUnity(order: order, power: power))
                }
            }
        }

        var seen = Set<String>()
        let deduped = expressions
            .map(\.canonicalized)
            .filter { seen.insert($0.canonicalKey).inserted }
            .map { expression -> CandidateBinding in
                let approx = expression.approximateValue
                let distance = max(abs(approx.re - target.re), abs(approx.im - target.im))
                return CandidateBinding(
                    variableID: variableID,
                    atomReplacements: [AtomReplacement(atomID: atom.id, expression: expression)],
                    provenance: CandidateProvenance(providerID: providerID, family: familyName(for: expression)),
                    complexity: complexity(for: expression),
                    numericDistance: CandidateDistance(distance),
                    canonicalKey: CandidateCanonicalKey(expression.canonicalKey)
                )
            }
            .sorted { lhs, rhs in
                if lhs.complexity != rhs.complexity {
                    return lhs.complexity < rhs.complexity
                }
                if lhs.numericDistance != rhs.numericDistance {
                    return lhs.numericDistance < rhs.numericDistance
                }
                return lhs.canonicalKey < rhs.canonicalKey
            }

        return deduped
    }

    private static func realExpressions(near value: Double) -> [ExactRealExpression] {
        var expressions: [ExactRealExpression] = [.rational(.zero)]

        let rationalDenominators = [1, 2, 3, 4, 6, 8, 12, 16]
        for denominator in rationalDenominators {
            let center = Int((value * Double(denominator)).rounded())
            for delta in -1...1 {
                expressions.append(.rational(Rational(center + delta, denominator)))
            }
        }

        let candidateRadicands = [2, 3, 5]
        let radicalDenominators = [1, 2, 3, 4, 6, 8, 12]
        for radicand in candidateRadicands {
            let root = Foundation.sqrt(Double(radicand))
            for denominator in radicalDenominators {
                let coefficient = Int((value * Double(denominator) / root).rounded())
                for delta in -1...1 {
                    expressions.append(
                        .canonicalRationalTimesSqrt(
                            coefficient: Rational(coefficient + delta, denominator),
                            radicand: radicand
                        )
                    )
                }
            }
        }

        var seen = Set<String>()
        return expressions
            .map(\.canonicalized)
            .filter { seen.insert($0.canonicalKey).inserted }
    }

    private static func scalarExpression(for real: ExactRealExpression) -> ExactScalarExpression {
        switch real.canonicalized {
        case let .rational(value):
            return .rational(value)
        case let .rationalTimesSqrt(coefficient, radicand):
            return .signedRationalTimesSqrt(coefficient: coefficient, radicand: radicand)
        }
    }

    private static func familyName(for expression: ExactScalarExpression) -> String {
        switch expression.canonicalized {
        case .rational:
            return "rational"
        case .signedRationalTimesSqrt:
            return "radical"
        case .complex:
            return "complex"
        case .rootOfUnity:
            return "rootOfUnity"
        }
    }

    private static func complexity(for expression: ExactScalarExpression) -> CandidateComplexityRank {
        switch expression.canonicalized {
        case let .rational(value):
            return CandidateComplexityRank(
                familyTier: 0,
                expressionNodeCount: 1,
                denominatorMagnitude: value.denominator,
                radicandMagnitude: 0,
                domainSeedTier: 0
            )
        case let .signedRationalTimesSqrt(coefficient, radicand):
            return CandidateComplexityRank(
                familyTier: 1,
                expressionNodeCount: 2,
                denominatorMagnitude: coefficient.denominator,
                radicandMagnitude: radicand,
                domainSeedTier: 0
            )
        case let .complex(real, imag):
            return CandidateComplexityRank(
                familyTier: 2,
                expressionNodeCount: expression.expressionNodeCount,
                denominatorMagnitude: max(denominatorMagnitude(in: real), denominatorMagnitude(in: imag)),
                radicandMagnitude: max(radicandMagnitude(in: real), radicandMagnitude(in: imag)),
                domainSeedTier: 0
            )
        case let .rootOfUnity(order, _):
            return CandidateComplexityRank(
                familyTier: 3,
                expressionNodeCount: 2,
                denominatorMagnitude: 1,
                radicandMagnitude: order,
                domainSeedTier: 0
            )
        }
    }

    private static func denominatorMagnitude(in expression: ExactRealExpression) -> Int {
        switch expression.canonicalized {
        case let .rational(value):
            return value.denominator
        case let .rationalTimesSqrt(coefficient, _):
            return coefficient.denominator
        }
    }

    private static func radicandMagnitude(in expression: ExactRealExpression) -> Int {
        switch expression.canonicalized {
        case .rational:
            return 0
        case let .rationalTimesSqrt(_, radicand):
            return radicand
        }
    }
}

private struct ScalarVerificationContext: Hashable, Sendable, Codable {
    let threshold: Double
    let verifierID: String
}
