"""
dtree._flip — EVPI and EVSI flip (value-of-information) operations.
Private module; do not import directly from user code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import (
    Branch,
    ChanceNode,
    DecisionNode,
    Prob,
    ValidationError,
)

if TYPE_CHECKING:
    from .core import DecisionTree


# ===========================================================================
# PUBLIC ENTRY POINT
# ===========================================================================

def run_flip(
    tree: "DecisionTree",
    decision_name: str,
    chance_name: str,
    test=None,
    likelihood=None,
) -> "DecisionTree":
    """
    Return a new tree placing perfect or sample information before the reference
    decision.

    EVPI  (test=None, likelihood=None):
        The true-state chance node becomes the top-level node. Each state branch
        leads to a copy of the decision subtree where the chance node carries a
        degenerate distribution (probability 1 on the known state).

    EVSI  (test + likelihood provided):
        The test/signal node is placed at the top with marginal probabilities
        (law of total probability). Each signal branch leads to a copy of the
        decision subtree where the chance node carries posterior probabilities
        computed via Bayes' theorem.

    Parameters
    ----------
    decision_name : name of the reference DecisionNode.
    chance_name   : name of the true-state ChanceNode that follows the decision.
    test          : ChanceNode whose branches are the test outcomes (EVSI only).
                    Branch labels and edge values are copied; probabilities are
                    replaced with computed marginals.
    likelihood    : P(test_outcome_j | true_state_i).
        list form : [[p00, p01, …], [p10, p11, …], …]
                    rows = true states (order must match chance_name.branches)
                    cols = test outcomes (order must match test.branches)
        dict form : {state_label: Prob([p0, p1, …]) or [p0, p1, …]}
                    keys = branch labels of chance_name; Prob allows SA on likelihoods
        None      : identity matrix — EVPI (perfect information)
    """
    from ._sensitivity import _find_node

    # ── 1. Locate nodes ──────────────────────────────────────────────────────
    decision_node = _find_node(tree.root, decision_name)
    if decision_node is None:
        raise ValidationError(
            f"DecisionNode '{decision_name}' not found in tree.", rule="node_not_found"
        )
    if not isinstance(decision_node, DecisionNode):
        raise ValidationError(
            f"'{decision_name}' is not a DecisionNode.", rule="wrong_node_type"
        )

    chance_node = _find_node(tree.root, chance_name)
    if chance_node is None:
        raise ValidationError(
            f"ChanceNode '{chance_name}' not found in tree.", rule="node_not_found"
        )
    if not isinstance(chance_node, ChanceNode):
        raise ValidationError(
            f"'{chance_name}' is not a ChanceNode.", rule="wrong_node_type"
        )
    if not _is_reachable(decision_node, chance_name):
        raise ValidationError(
            f"ChanceNode '{chance_name}' is not reachable from DecisionNode '{decision_name}'.",
            rule="node_not_reachable",
        )

    # A name can legitimately repeat under sibling decision branches (e.g. the
    # same risk follows either action, just with action-dependent payoffs) --
    # every occurrence downstream of decision_node must carry the flipped
    # posterior, not just the first one _find_node happens to hit. Checked
    # against the pre-copy tree so a genuine modeling inconsistency (two
    # same-named nodes that don't actually represent the same uncertainty)
    # is caught before any copying happens.
    all_chance_occurrences = _find_all_nodes(decision_node, chance_name)
    reference_probs = list(all_chance_occurrences[0].probs.base)
    for occurrence in all_chance_occurrences[1:]:
        if occurrence.probs is None or occurrence.probs.base is None or \
                list(occurrence.probs.base) != reference_probs:
            raise ValidationError(
                f"Multiple ChanceNodes named '{chance_name}' exist under "
                f"'{decision_name}', but their base probabilities don't match "
                f"({reference_probs} vs {occurrence.probs.base if occurrence.probs else None}). "
                "flip() treats same-named nodes as one shared uncertainty -- "
                "if these are genuinely different risks, give them different names.",
                node=chance_name, rule="ambiguous_chance_node",
            )

    # ── 2. Priors ─────────────────────────────────────────────────────────────
    if chance_node.probs is None or chance_node.probs.base is None:
        raise ValidationError(
            f"ChanceNode '{chance_name}' has no base probabilities.", rule="missing_probs"
        )
    priors = list(chance_node.probs.base)
    n_states = len(chance_node.branches)

    # ── 3. Parse likelihood ──────────────────────────────────────────────────
    lk = _parse_likelihood(likelihood, chance_node, test, n_states)
    n_outcomes = len(lk[0])

    # ── 4. Test branch labels ─────────────────────────────────────────────────
    if test is None:
        # EVPI: test outcomes are the true states themselves. The new
        # top-level branches only reveal which state holds -- any value or
        # time on chance_node's own branches is already accumulated inside
        # the copied decision subtree below (that same chance node, now
        # forced, keeps its original branch value), so copying it here too
        # would double-count it. Only correct by accident when that value
        # happens to be 0, e.g. Freemark/Texaco's convention of attaching
        # payoffs to LeafNodes instead of chance-node branches.
        test_labels  = [b.label for b in chance_node.branches]
        test_values  = [0] * n_states
        test_times   = [0] * n_states
    else:
        if len(test.branches) != n_outcomes:
            raise ValidationError(
                f"test node has {len(test.branches)} branches but likelihood "
                f"has {n_outcomes} columns.",
                rule="likelihood_shape_mismatch",
            )
        test_labels = [b.label for b in test.branches]
        test_values = [b.value for b in test.branches]
        test_times  = [b.time  for b in test.branches]

    # ── 5. Bayes ──────────────────────────────────────────────────────────────
    marginals, posteriors = _bayes(priors, lk)

    # ── 6. Build flipped subtree ──────────────────────────────────────────────
    # Name for top node: test node name (EVSI) or original chance name (EVPI)
    top_name = test.name if test is not None else chance_name
    # Inner copies get a distinct name to avoid path-level name collisions
    inner_name = chance_name + "_posterior"

    top_node = ChanceNode(top_name)
    top_node.probs = Prob(base=marginals)

    for j in range(n_outcomes):
        # Deep-copy the decision subtree independently (fresh event_ids)
        decision_copy = decision_node.copy(deep=True, independent=True)

        # Update every occurrence of the chance node inside the copy, not
        # just the first match -- a name shared across sibling decision
        # branches (same risk, action-dependent payoff) means more than one
        # distinct node instance needs the same posterior applied.
        chance_copies = _find_all_nodes(decision_copy, chance_name)
        if not chance_copies:
            raise ValidationError(
                f"ChanceNode '{chance_name}' not found in decision subtree copy.",
                rule="internal_error",
            )
        for chance_copy in chance_copies:
            chance_copy.name = inner_name
            chance_copy.probs = Prob(base=posteriors[j])

        top_node.branches.append(Branch(
            label=test_labels[j],
            value=test_values[j],
            time=test_times[j],
            child=decision_copy,
        ))

    # ── 7. Graft into full tree if decision is not the root ───────────────────
    if decision_node is tree.root:
        new_root = top_node
    else:
        new_root = tree.root.copy(deep=True, independent=True)
        parent_branch = _find_parent_branch(new_root, decision_name)
        if parent_branch is None:
            raise ValidationError(
                f"Could not locate parent of '{decision_name}' in full tree copy.",
                rule="internal_error",
            )
        parent_branch.child = top_node

    return type(tree)._from_root(new_root, tree.settings)


# ===========================================================================
# HELPERS
# ===========================================================================

def _parse_likelihood(
    likelihood,
    chance_node: ChanceNode,
    test,
    n_states: int,
) -> list[list[float]]:
    """
    Normalise likelihood into lk[i][j] = P(test_outcome_j | true_state_i).

    None     → identity matrix (EVPI)
    list     → passed through with shape check
    dict     → {state_label: Prob or list}, rows extracted in branch order
    """
    if likelihood is None:
        # Identity matrix: test outcome j = "observe state j directly"
        return [[1.0 if i == j else 0.0 for j in range(n_states)] for i in range(n_states)]

    if isinstance(likelihood, dict):
        rows: list[list[float]] = []
        for branch in chance_node.branches:
            lbl = branch.label
            if lbl not in likelihood:
                raise ValidationError(
                    f"likelihood dict is missing key '{lbl}' "
                    f"(branch of ChanceNode '{chance_node.name}').",
                    rule="likelihood_missing_key",
                )
            v = likelihood[lbl]
            rows.append(list(v.base) if isinstance(v, Prob) else list(v))
        _validate_lk_shape(rows, n_states, chance_node.name)
        return rows

    # List form
    lk = [list(row) for row in likelihood]
    _validate_lk_shape(lk, n_states, chance_node.name)
    return lk


def _validate_lk_shape(lk: list[list[float]], n_states: int, name: str) -> None:
    if len(lk) != n_states:
        raise ValidationError(
            f"likelihood has {len(lk)} rows but ChanceNode '{name}' has {n_states} branches.",
            rule="likelihood_shape_mismatch",
        )
    n_cols = len(lk[0])
    for i, row in enumerate(lk):
        if len(row) != n_cols:
            raise ValidationError(
                f"likelihood row {i} has {len(row)} values but row 0 has {n_cols}.",
                rule="likelihood_shape_mismatch",
            )
        total = sum(row)
        if abs(total - 1.0) > 1e-6:
            raise ValidationError(
                f"likelihood row {i} (state '{name}') sums to {total:.6f}, not 1.0.",
                rule="likelihood_not_normalized",
            )


def _bayes(
    priors: list[float],
    lk: list[list[float]],
) -> tuple[list[float], list[list[float]]]:
    """
    priors[i]   = P(state_i)
    lk[i][j]    = P(test_j | state_i)

    Returns:
        marginals[j]      = P(test_j)
        posteriors[j][i]  = P(state_i | test_j)
    """
    n_states   = len(priors)
    n_outcomes = len(lk[0])

    marginals = [
        sum(priors[i] * lk[i][j] for i in range(n_states))
        for j in range(n_outcomes)
    ]

    posteriors = []
    for j in range(n_outcomes):
        m = marginals[j]
        if m < 1e-15:
            # Zero-probability test outcome: fall back to priors to keep tree valid
            posteriors.append(list(priors))
        else:
            posteriors.append([priors[i] * lk[i][j] / m for i in range(n_states)])

    return marginals, posteriors


def _is_reachable(root, target_name: str) -> bool:
    """Return True if a node named target_name exists in root's subtree."""
    from ._sensitivity import _find_node
    return _find_node(root, target_name) is not None


def _find_all_nodes(root, name: str) -> list:
    """Walk the tree (DAG-safe) and return every node whose .name matches,
    in the order encountered. A name shared across sibling branches by two
    distinct node objects (not just one object referenced twice) returns
    each of them; a name reached twice via the same shared object returns
    it only once, since the visited set is keyed by identity."""
    found: list = []
    _find_all_walk(root, name, set(), found)
    return found


def _find_all_walk(node, name: str, visited: set, found: list) -> None:
    if node is None or id(node) in visited:
        return
    visited.add(id(node))
    if node.name == name:
        found.append(node)
    for b in getattr(node, "branches", []):
        _find_all_walk(b.child, name, visited, found)


def _find_parent_branch(root, child_name: str) -> "Branch | None":
    """
    Depth-first search for a branch whose .child.name == child_name.
    Returns the Branch object so the caller can redirect .child.
    DAG-safe via visited set on node ids.
    """
    return _find_parent_walk(root, child_name, set())


def _find_parent_walk(node, child_name: str, visited: set):
    if node is None or id(node) in visited:
        return None
    visited.add(id(node))
    for b in getattr(node, "branches", []):
        if b.child is not None and b.child.name == child_name:
            return b
        result = _find_parent_walk(b.child, child_name, visited)
        if result is not None:
            return result
    return None
