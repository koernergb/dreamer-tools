"""Pinned source instrumentation for DreamerV3 policy inspection."""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from types import ModuleType
from typing import Any

POLICY_SOURCE_SHA256 = (
    "f547f5a69c799483aab2119a6cae60415fd1ce738a90915e3ed85135928b3af8"
)

_INSTRUMENTATION = """
prior_logits = self.dyn._prior(feat['deter'])
_, _, extract_recons = self.dec({}, feat, reset, **kw)
extract_policy = lambda x: sample(self.pol(self.feat2tensor(x), 1))
_, extract_imgfeat, _ = self.dyn.imagine(
    dyn_carry, extract_policy, self.config.imag_length, training=False)
extract_imgreset = jnp.zeros(
    (reset.shape[0], self.config.imag_length), dtype=bool)
_, _, extract_imaginations = self.dec(
    {}, extract_imgfeat, extract_imgreset, training=False)
out['extract/h'] = feat['deter']
out['extract/z'] = feat['stoch']
out['extract/prior_logits'] = prior_logits
out['extract/posterior_logits'] = feat['logit']
extract_tensor = self.feat2tensor(feat)
out['extract/reward_prediction'] = self.rew(extract_tensor, 1).pred()
out['extract/continuation_prediction'] = self.con(extract_tensor, 1).prob(1)
for extract_key, extract_value in extract_recons.items():
    if extract_key in self.dec.imgkeys:
        out[f'extract/reconstruction/{extract_key}'] = extract_value.pred()
for extract_key, extract_value in extract_imaginations.items():
    if extract_key in self.dec.imgkeys:
        out[f'extract/imagination/{extract_key}'] = extract_value.pred()
"""


def instrument_policy(agent_module: ModuleType) -> None:
    """Add extraction outputs to the exact pinned policy before JAX compilation."""
    cls = agent_module.Agent
    source = textwrap.dedent(inspect.getsource(cls.policy))
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != POLICY_SOURCE_SHA256:
        raise RuntimeError(
            "Pinned DreamerV3 policy source changed; refusing unsafe instrumentation "
            f"({digest})"
        )
    tree = ast.parse(source)
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef):
        raise RuntimeError("Could not locate DreamerV3 policy function")
    insertion = next(
        (
            index
            for index, node in enumerate(function.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "carry"
                for target in node.targets
            )
        ),
        None,
    )
    if insertion is None:
        raise RuntimeError("Could not locate DreamerV3 policy output boundary")
    statements = ast.parse(_INSTRUMENTATION).body
    function.body[insertion:insertion] = statements
    ast.fix_missing_locations(tree)
    namespace: dict[str, Any] = dict(vars(agent_module))
    exec(
        compile(tree, filename="<dreamer-tools-instrumented-policy>", mode="exec"),
        namespace,
    )
    cls.policy = namespace["policy"]
    cls.policy_keys = property(lambda self: "^(enc|dyn|dec|pol|rew|con)/")
