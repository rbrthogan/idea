import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_checkpoint_cache_cannot_enable_rating_or_write_a_repaired_evolution():
    source = (ROOT / "idea/static/js/viewer.js").read_text()
    start = source.index("function overlayHistoryItemsWithCachedEvolution")
    helper = source[start:source.index("async function loadHistory()", start)]
    script = """
const cached = { evolution_id: 'legacy', current_generation: 4, history: [[{}], [{}]] };
function normalizeCachedEvolutionState(value) { return value; }
function getCachedEvolutionState() { return cached; }
function getCachedEvolutionId(value) { return value.evolution_id; }
function cachedEvolutionIsNewer() { throw new Error('Legacy cache must not replace the source'); }
function repairPersistedEvolutionFromClientCache() { throw new Error('Legacy source is read-only'); }
""" + helper + """
const sourceItem = { id: 'legacy', type: 'legacy_checkpoint', can_rate: false, generations: 1 };
const result = overlayHistoryItemsWithCachedEvolution([sourceItem]);
process.stdout.write(JSON.stringify({ unchanged: result[0] === sourceItem, can_rate: result[0].can_rate }));
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=10)
    assert json.loads(result.stdout) == {"unchanged": True, "can_rate": False}


def test_opening_legacy_checkpoint_fetches_the_read_only_route_without_cache_repair():
    source = (ROOT / "idea/static/js/viewer.js").read_text()
    start = source.index("async function loadHistoryItemData")
    helper = source[start:source.index("async function handleHistoryResume", start)]
    script = """
const requests = [];
global.fetch = async url => {
    requests.push(url);
    return { ok: true, json: async () => ({ id: 'legacy', data: {} }) };
};
const errors = [];
global.console = { error: (...args) => errors.push(args.map(String)) };
async function repairCurrentEvolutionIfNewer() { throw new Error('Legacy source is read-only'); }
""" + helper + """
(async () => {
    await loadHistoryItemData({ id: 'legacy', type: 'legacy_checkpoint' });
    process.stdout.write(JSON.stringify({ requests, errors }));
})();
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=10)
    assert json.loads(result.stdout) == {"requests": ["/api/evolution/legacy"], "errors": []}


def run_repair_scenario(scenario):
    source = (ROOT / "idea/static/js/viewer.js").read_text()
    start = source.index("const repairedEvolutionCacheIds")
    helpers = source[start:source.index("async function repairCurrentEvolutionIfNewer", start)]
    script = """
let isEvolutionRunning = false;
let currentEvolutionData = null;
let currentEvolutionId = null;
const warnings = [];
global.console = { warn: (...values) => warnings.push(values.map(String)) };
const requests = [];
let responses = [];
global.fetch = async (url, options) => {
    requests.push({ url, data: JSON.parse(options.body).data });
    return responses.shift() || { ok: true, status: 200 };
};
function cached(size, running = false) {
    return { evolution_id: 'run-1', history: Array.from({ length: size }, (_, index) => [{ id: `idea-${index}` }]), is_running: running };
}
""" + helpers + "\n(async () => {" + scenario + """
})().catch(error => { process.stderr.write(error.stack); process.exit(1); });
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=10)
    return json.loads(result.stdout)


def test_active_repairs_wait_for_confirmed_stop_and_use_the_latest_history():
    result = run_repair_scenario("""
await repairPersistedEvolutionFromClientCache(cached(2, true));
isEvolutionRunning = true;
await repairPersistedEvolutionFromClientCache(cached(4));
await repairPersistedEvolutionFromClientCache(cached(1, true));
await retryDeferredEvolutionRepairs({ is_running: true });
const whileRunning = requests.length;
isEvolutionRunning = false;
await retryDeferredEvolutionRepairs({ ...cached(3), is_stopped: true });
await repairPersistedEvolutionFromClientCache(cached(4));
process.stdout.write(JSON.stringify({ whileRunning, requests, warnings, deferred: deferredEvolutionCacheStates.size }));
""")
    assert result["whileRunning"] == 0
    assert len(result["requests"]) == 1
    payload = result["requests"][0]["data"]
    assert len(payload["history"]) == 4
    assert payload["is_running"] is False
    assert payload["status"] == "paused"
    assert result["deferred"] == 0
    assert result["warnings"] == []


def test_worker_conflict_is_silent_and_does_not_resubmit_until_stop_is_confirmed():
    result = run_repair_scenario("""
responses = [{ ok: false, status: 409, json: async () => ({ status: 'conflict', message: 'Worker owns active evolution' }) }];
await repairPersistedEvolutionFromClientCache(cached(2));
await repairPersistedEvolutionFromClientCache(cached(3));
await retryDeferredEvolutionRepairs(null);
await retryDeferredEvolutionRepairs({ is_running: true });
const beforeStopped = requests.length;
await retryDeferredEvolutionRepairs({ ...cached(3), status: 'complete' });
await repairPersistedEvolutionFromClientCache(cached(3));
process.stdout.write(JSON.stringify({ beforeStopped, requests, warnings, deferred: deferredEvolutionCacheStates.size }));
""")
    assert result["beforeStopped"] == 1
    assert len(result["requests"]) == 2
    assert len(result["requests"][1]["data"]["history"]) == 3
    assert result["requests"][1]["data"]["status"] == "complete"
    assert result["deferred"] == 0
    assert result["warnings"] == []
