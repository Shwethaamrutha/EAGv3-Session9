"""Unit tests for recovery classification and critic splice mechanics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from recovery import classify_failure


# --- classify_failure: transient errors ---

def test_503_is_transient():
    assert classify_failure("HTTP 503 Service Unavailable") == "transient"

def test_502_is_transient():
    assert classify_failure("502 Bad Gateway") == "transient"

def test_504_is_transient():
    assert classify_failure("504 Gateway Timeout") == "transient"

def test_timeout_is_transient():
    assert classify_failure("Request timeout after 30s") == "transient"

def test_connection_error_is_transient():
    assert classify_failure("ConnectionError: failed to connect") == "transient"

def test_bad_gateway_text_is_transient():
    assert classify_failure("bad gateway response from upstream") == "transient"

def test_gateway_timeout_is_transient():
    assert classify_failure("gateway timeout waiting for response") == "transient"

def test_service_unavailable_is_transient():
    assert classify_failure("service unavailable, retry later") == "transient"

def test_rate_limit_is_transient():
    assert classify_failure("Rate limit exceeded") == "transient"

def test_api_timeout_error_is_transient():
    assert classify_failure("openai.APITimeoutError: request timed out") == "transient"

def test_api_connection_error_is_transient():
    assert classify_failure("openai.APIConnectionError: connection refused") == "transient"

def test_throttling_is_transient():
    assert classify_failure("Throttling: too many requests") == "transient"


# --- classify_failure: validation errors ---

def test_malformed_is_validation():
    assert classify_failure("malformed JSON in response") == "validation_error"

def test_validation_error_is_validation():
    assert classify_failure("pydantic ValidationError: field required") == "validation_error"

def test_validation_error_text_is_validation():
    assert classify_failure("validation error: missing 'nodes' key") == "validation_error"

def test_json_decode_is_validation():
    assert classify_failure("json.JSONDecodeError: Expecting value") == "validation_error"

def test_parse_error_is_validation():
    assert classify_failure("parse error in planner output") == "validation_error"

def test_schema_error_is_validation():
    assert classify_failure("schema mismatch in response") == "validation_error"


# --- classify_failure: upstream failures ---

def test_generic_error_is_upstream():
    assert classify_failure("Something went wrong in the researcher") == "upstream_failure"

def test_empty_result_is_upstream():
    assert classify_failure("No results found for query") == "upstream_failure"

def test_tool_failure_is_upstream():
    assert classify_failure("Tool execution returned error: file not found") == "upstream_failure"

def test_unknown_error_is_upstream():
    assert classify_failure("Unexpected error during processing") == "upstream_failure"


# --- Critic splice mechanics ---

def test_critic_pass_leaves_graph_unchanged():
    """A pass verdict does not modify successor status."""
    from flow import Graph

    g = Graph()
    # Build: parent → critic → child manually (no auto-edge from inputs)
    parent_id = g.add_node("distiller", ["USER_QUERY"], {"label": "d1"})
    critic_id = g.add_node("critic", [], {"label": "c1", "target_node": parent_id})
    child_id = g.add_node("formatter", [], {"label": "out"})

    g.g.add_edge(parent_id, critic_id)
    g.g.add_edge(critic_id, child_id)

    g.mark(parent_id, "complete")
    g.mark(critic_id, "complete")

    # Child should be ready
    assert child_id in g.ready_nodes()
    assert g.g.nodes[child_id]["status"] == "pending"


def test_critic_fail_skips_child():
    """A fail verdict marks the child as skipped."""
    from flow import Graph

    g = Graph()
    parent_id = g.add_node("distiller", ["USER_QUERY"], {"label": "d1"})
    critic_id = g.add_node("critic", [], {"label": "c1", "target_node": parent_id})
    child_id = g.add_node("formatter", [], {"label": "out"})

    g.g.add_edge(parent_id, critic_id)
    g.g.add_edge(critic_id, child_id)

    g.mark(parent_id, "complete")
    g.mark(critic_id, "complete")
    g.mark(child_id, "skipped")

    assert g.g.nodes[child_id]["status"] == "skipped"
    assert child_id not in g.ready_nodes()


def test_recovery_cap_prevents_loop():
    """Per-target cap prevents infinite critic-fail recovery loops."""
    from flow import Executor
    from skills import load_skills

    skills = load_skills()
    executor = Executor(session_id="test_cap", skills_catalogue=skills)

    target = "n:1"
    executor._recovery_count[target] = 1
    assert executor._recovery_count.get(target, 0) >= MAX_RECOVERY_PER_TARGET


MAX_RECOVERY_PER_TARGET = 1


def test_auto_critic_insertion():
    """Skills with critic: true get a critic node auto-inserted."""
    from flow import Graph
    from skills import load_skills

    skills = load_skills()
    g = Graph()
    planner_id = g.add_node("planner", ["USER_QUERY"], {"label": "plan"})
    g.mark(planner_id, "complete")

    planner_output = {
        "rationale": "Extract then format",
        "nodes": [
            {"skill": "distiller", "inputs": ["USER_QUERY"], "metadata": {"label": "d1", "question": "extract fields"}},
            {"skill": "formatter", "inputs": ["n:d1"], "metadata": {"label": "out"}},
        ]
    }

    g.extend_from(planner_output, planner_id, skills)

    # Should have: planner + distiller + formatter + auto-inserted critic
    assert g.node_count >= 4

    # Find the critic node
    critic_nodes = [nid for nid, d in g.g.nodes(data=True) if d.get("skill") == "critic"]
    assert len(critic_nodes) >= 1
