""" 
# version 1.0
# for ubaru class

"""
import numpy as np
from scipy.sparse import issparse, spmatrix
from collections import defaultdict, Iterator
from itertools import combinations, chain
from functools import reduce

_FP_TREE_EMPTY = (None, [])
_BUCKETING_FEW_ITEMS = 10

class OneHot:
    """
    Encode discrete Orange.data.Table into a 2D array of binary attributes.
    """
    @staticmethod
    def encode(table, include_class=False):
        """
        Return a tuple of
        (bool (one hot) ndarray, {col: (variable_index, value_index)} mapping)

        If the input table is sparse, a list of nonzero column indices
        per row (LIL rows) is returned instead of the one-hot ndarray.
        """
        X, encoded, mapping = table.X, [], {}
        if issparse(X):
            encoded = X.tolil().rows.tolist()
            for i, var in enumerate(table.domain.attributes):
                mapping[i] = i, 0
        else:
            for i, var in enumerate(table.domain.attributes):
                if not var.is_discrete: continue
                for j, val in enumerate(var.values):
                    mapping[len(mapping)] = i, j
                    encoded.append(X[:, i] == j)

        if include_class and table.domain.has_discrete_class:
            i, var = len(table.domain.attributes), table.domain.class_var
            for j, val in enumerate(var.values):
                mapping[len(mapping)] = i, j
                if issparse(X):
                    for row in encoded:
                        row.append(i + j)
                else:
                    encoded.append(table.Y == j)

        if not issparse(X):
            encoded = np.column_stack(encoded) if encoded else None
        return encoded, mapping

    @staticmethod
    def decode(itemset, table, mapping):
        """Yield sorted (item, variable, value) tuples (one for each item)"""
        attributes = table.domain.attributes
        for item in itemset:
            ivar, ival = mapping[item]
            var = attributes[ivar] if ivar < len(attributes) else table.domain.class_var
            yield item, var, (var.values[ival] if var.is_discrete else 0)
            
class _Node(dict):
    def __init__(self, item=None, parent=None, count=None):
        self.item = item
        self.parent = parent
        self.count = count
            
def frequent_itemsets(X, min_support=.2):
    """
    Generator yielding frequent itemsets from database X.

    Parameters
    ----------
    X : list or numpy.ndarray or scipy.sparse.spmatrix or iterator
        The database of transactions where each transaction is a collection
        of integer items. If `numpy.ndarray`, the items are considered to be
        indices of non-zero columns.
    min_support : float or int
        If float in range (0, 1), percent of minimal support for itemset to
        be considered frequent. If int > 1, the absolute number of instances.
        For example, general iterators don't have defined length, so you need
        to pass the absolute minimal support as int.

    Yields
    ------
    itemset: frozenset
        Iteratively yields all itemsets (as frozensets of item indices) with
        support greater or equal to specified `min_support`.
    support: int
        Itemset's support as number of instaances.

    Examples
    --------
    Have a database of 50 transactions, 100 possible items:

    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X = np.random.random((50, 100)) > .9

    Convert it to sparse so we show this type is supported:

    >>> from scipy.sparse import lil_matrix  # other types would convert to LIL anyway
    >>> X = lil_matrix(X)

    Count the number of itemsets of at least two items with support greater
    than 4%:

    >>> sum(1 for itemset, support in frequent_itemsets(X, .05)
    ...     if len(itemset) >= 2)
    72

    Let's get all the itemsets with at least 20% support:

    >>> gen = frequent_itemsets(X, .2)
    >>> gen
    <generator object ...>

    >>> itemsets = list(gen)
    >>> itemsets
    [(frozenset({4}), 11), (frozenset({25}), 10)]

    We get the same result by specifying the support as absolute number:

    >>> list(frequent_itemsets(X, 10)) == itemsets
    True

    So the items '4' and '25' (fifth and twenty sixth columns of X) are the
    only items (and itemsets) that appear 10 or more times. Let's check this:

    >>> (X.sum(axis=0) >= 10).nonzero()[1]
    array([ 4, 25])

    Conclusion: Given databases of uniformly distributed random data,
    there's not much to work with.
    """
    if not isinstance(X, (np.ndarray, spmatrix, list, Iterator)):
        raise TypeError('X must be (sparse) array of boolean values, or'
                        'list of lists of hashable items, or iterator')
    if not (isinstance(min_support, int) and min_support > 0 or
            isinstance(min_support, float) and 0 < min_support <= 1):
        raise ValueError('min_support must be an integer number of instances,'
                         'or a percent fraction in (0, 1]')

    min_support *= (1 if isinstance(min_support, int) else
                    len(X) if isinstance(X, list) else
                    X.shape[0])
    min_support = max(1, int(np.ceil(min_support)))

    if issparse(X):
        X = X.tolil().rows
    elif isinstance(X, np.ndarray):
        X = (t.nonzero()[-1] for t in X)

    db = ((1, transaction) for transaction in X)  # 1 is initial item support
    tree, itemsets = _fp_tree(db, min_support)
    if itemsets:
        yield from itemsets
    if tree:
        yield from _fp_growth(tree, frozenset(), min_support)
        
def _fp_tree(db, min_support):
    """
    FP-tree construction ([1] § 2.1, Algorithm 1).

    If frequent items in db are determined to be less than threshold,
    "bucketing" [2] is used instead.

    Returns
    -------
    tuple
        (FP-tree, None) or (None, list of frequent itemsets with support)
    """
    if not isinstance(db, list): db = list(db)

    if not db:
        return _FP_TREE_EMPTY

    # Used to count item support so it can be reported when generating itemsets
    item_support = defaultdict(int)
    # Used for ordering transactions' items for "optimally" "compressed" tree
    node_support = defaultdict(int)
    for count, transaction in db:
        for item in transaction:
            item_support[item] += count
            node_support[item] += 1
    # Only ever consider items that have min_support
    frequent_items = {item
                      for item, support in item_support.items()
                      if support >= min_support}

    # Short-circuit, if possible
    n_items = len(frequent_items)
    if 0 == n_items:
        return _FP_TREE_EMPTY
    if 1 == n_items:
        item = frequent_items.pop()
        return None, ((frozenset({item}), item_support[item]),)
    if n_items <= _BUCKETING_FEW_ITEMS:
        return None, ((frozenset(itemset), support)
                      for itemset, support in _bucketing_count(db, frequent_items, min_support))

    # "The items [...] should be ordered in the frequency descending order of
    # node occurrence of each item instead of its support" ([1], p. 12, bottom)
    sort_index = {item: i
                  for i, item in
                      enumerate(sorted(frequent_items,
                                       key=node_support.__getitem__,
                                       reverse=True))}.__getitem__
    # Only retain frequent items and sort them
    db = ((count, sorted(frequent_items.intersection(transaction),
                         key=sort_index))
          for count, transaction in db)

    root = _Node()
    node_links = defaultdict(list)
    for count, transaction in db:
        T = root
        for item in transaction:
            T = _fp_tree_insert(item, T, node_links, count)
    # Sorted support-descending (in reverse because popping from the back for efficiency)
    root.node_links = sorted(node_links.items(), key=lambda i: -sort_index(i[0]))
    return root, None

def _fp_tree_insert(item, T, node_links, count):
    """ Insert item into _Node-tree T and return the new node """
    node = T.get(item)
    if node is None:
        node = T[item] = _Node(item, T, count)
        node_links[item].append(node)
    else:  # Node for this item already in T, just inc its count
        node.count += count
    return node

def _fp_growth(tree, alpha, min_support):
    """ FP-growth ([1], § 3.3, Algorithm 2). """
    # Single prefix path optimization ([1] § 3.1)
    P, Q = _single_prefix_path(tree) if len(tree) == 1 else ([], tree)
    # Return P×Q
    yield from _freq_patterns_single(P, alpha, min_support)
    for itemsetQ, supportQ in _freq_patterns_multi(Q, alpha, min_support):
        yield itemsetQ, supportQ
        for itemsetP, supportP in _freq_patterns_single(P, alpha, min_support):
            yield itemsetQ | itemsetP, supportQ
            
def _freq_patterns_single(P, alpha, min_support):
    """ Yield subsets of P as (frequent itemset, support) """
    for itemset in _powerset(P):
        yield alpha.union(i[0] for i in itemset), itemset[-1][1]
        
def _powerset(lst):
    """
    >>> list(_powerset([1, 2, 3]))
    [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    return chain.from_iterable(combinations(lst, r)
                               for r in range(1, len(lst) + 1))

def _freq_patterns_multi(Q, alpha, min_support):
    """ Mine multi-path FP-tree """
    for item, nodes in reversed(Q.node_links):
        support = sum(n.count for n in nodes)
        beta = alpha.union({item})
        yield beta, support
        tree, got_itemsets = _fp_tree(_prefix_paths(Q, nodes), min_support)
        if got_itemsets:
            for itemset, support in got_itemsets:
                yield beta.union(itemset), support
        elif tree is not None:
            yield from _fp_growth(tree, beta, min_support)
            
def _prefix_paths(tree, nodes):
    """ Generate all paths of tree leading to all item nodes """
    for node in nodes:
        path = []
        support = node.count
        node = node.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent
        if path:
            yield support, path
            
def _bucketing_count(db, frequent_items, min_support):
    """
    Bucket counting (bucketing) optimization for databases where few items
    are frequent ([2] § 5).
    """
    # Forward and inverse mapping of frequent_items to [0, n_items)
    inv_map = dict(enumerate(frequent_items)).__getitem__
    fwd_map = {v: k for k, v in inv_map.__self__.items()}.__getitem__
    # Project transactions
    k = len(frequent_items)
    buckets = [0] * 2**k
    for count, transaction in db:
        set_bits = (fwd_map(i) for i in frequent_items.intersection(transaction))
        tid = reduce(lambda a, b: a | 1 << b, set_bits, 0)
        buckets[tid] += count
    # Aggregate bucketing counts ([2], Figure 5)
    for i in range(0, k):
        i = 2**i
        for j in range(2**k):
            if j & i == 0:
                buckets[j] += buckets[j + i]
    # Announce results
    buckets = enumerate(buckets)
    next(buckets)  # Skip 000...0
    for tid, count in buckets:
        if count >= min_support:
            yield frozenset(inv_map(i) for i, b in enumerate(reversed(bin(tid))) if b == '1'), count

def association_rules(itemsets, min_confidence, itemset=None):
    """
    Generate association rules ([3] § 12.3) from dict of itemsets' supports
    (from :obj:`frequent_itemsets()`). If `itemset` is provided, only generate
    its rules.

    Parameters
    ----------
    itemsets: dict
        A `dict` mapping itemsets to their supports. Can be generated by
        feeding the output of `frequent_itemsets()` to `dict` constructor.
    min_confidence: float
        Confidence percent. Defined as `itemset_support / antecedent_support`.
    itemset: frozenset
        Itemset the association rules of which we are interested in.

    Yields
    ------
    antecedent: frozenset
        The LHS of the association rule.
    consequent: frozenset
        The RHS of the association rule.
    support: int
        The number of instances supporting (containing) this rule.
    confidence: float
        ``total_support / lhs_support``.

    Examples
    --------
    >>> np.random.seed(0)
    >>> N = 100
    >>> X = np.random.random((N, 100)) > .9

    Find all itemsets with at least 5% support:

    >>> itemsets = dict(frequent_itemsets(X, .05))
    >>> len(itemsets)
    116

    Generate all association rules from these itemsets with minimum
    50% confidence:

    >>> rules = association_rules(itemsets, .5)
    >>> rules
    <generator object ...>
    >>> rules = list(rules)
    >>> len(rules)
    7
    >>> rules
    [(frozenset({36}), frozenset({25}), 5, 0.55...),
     (frozenset({63}), frozenset({58}), 5, 0.5),
     ...
     (frozenset({30}), frozenset({32}), 5, 0.55...),
     (frozenset({75}), frozenset({98}), 5, 0.5)]

    Or only the rules for a particular itemset:

    >>> list(association_rules(itemsets, .3, frozenset({75, 98})))
    [(frozenset({75}), frozenset({98}), 5, 0.5),
     (frozenset({98}), frozenset({75}), 5, 0.45...)]

    """
    assert (isinstance(itemsets, dict) and
            isinstance(next(iter(itemsets), frozenset()), frozenset))
    assert 0 < min_confidence <= 1
    from_itemsets = (itemset,) if itemset else sorted(itemsets, key=len, reverse=True)
    for itemset in from_itemsets:
        support = itemsets[itemset]
        for item in itemset:
            right = frozenset({item})
            yield from _association_rules(
                itemset - right, right,
                item, support, min_confidence, itemsets)
            
def _association_rules(left, right, last_item, support, min_confidence, itemsets):
    if not left: return
    confidence = support / itemsets[left]
    if confidence >= min_confidence:
        yield left, right, support, confidence
        for item in left:
            if item > last_item: continue  # This ensures same rules aren't visited twice
            yield from _association_rules(
                left - {item}, right | {item},
                item, support, min_confidence, itemsets)
            
def rules_stats(rules, itemsets, n_examples):
    """
    Generate additional stats for rules generated by :obj:`association_rules()`.

    Parameters
    ----------
    rules: iterable
        Rules as output by `association_rules()`.
    itemsets: dict
        The itemsets as obtained by `dict(frequent_itemsets(...))`.
    n_examples: int
        The total number of instances (for calculating coverage, lift,
        and leverage).

    Yields
    ------
    atecedent: frozenset
        The LHS of the association rule.
    consequent: frozenset
        The RHS of the association rule.
    support: int
        Support as an absolute number of instances.
    confidence: float
        The confidence percent, calculated as: ``total_support / lhs_rupport``.
    coverage: float
        Calculated as: ``lhs_support / n_examples``
    strength: float
        Calculated as: ``rhs_support / lhs_examples``
    lift: float
        Calculated as: ``n_examples * total_support / lhs_support / rhs_support``
    leverage: float
        Calculated as: ``(total_support * n_examples - lhs_support * rhs_support) / n_examples**2``

    Examples
    --------
    >>> N = 30
    >>> X = np.random.random((N, 50)) > .9
    >>> itemsets = dict(frequent_itemsets(X, .1))
    >>> rules = association_rules(itemsets, .6)
    >>> list(rules_stats(rules, itemsets, N))
    [(frozenset({15}), frozenset({0}), 3, 0.75, 0.13..., 1.5, 3.75, 0.073...),
     (frozenset({47}), frozenset({22}), 3, 0.6, 0.16..., 1.4, 2.57..., 0.061...),
     (frozenset({27}), frozenset({22}), 4, 0.66..., 0.2, 1.16..., 2.85..., 0.086...),
     (frozenset({19}), frozenset({22}), 3, 0.6, 0.16..., 1.4, 2.57..., 0.061...)]

    """
    assert (isinstance(itemsets, dict) and
            isinstance(next(iter(itemsets), frozenset()), frozenset))
    assert n_examples > 0
    for left, right, support, confidence in rules:
        l_support, r_support = itemsets[left], itemsets[right]
        coverage = l_support / n_examples
        strength = r_support / l_support
        lift = n_examples * confidence / r_support
        leverage = (support*n_examples - l_support*r_support) / n_examples**2
        yield (left, right, support, confidence,
               coverage, strength, lift, leverage)
        
def _single_prefix_path(root):
    """ Return (single-prefix path, rest of tree with new root) """
    path = []
    tree = root
    node_links = root.node_links
    while len(tree) == 1:
        tree = next(iter(tree.values()))
        path.append((tree.item, tree.count))
        node_links.pop()
    tree.parent, tree.item, tree.node_links = None, None, node_links
    return path, tree