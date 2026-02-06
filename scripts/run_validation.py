"""
Backend Validation Script.
Runs all tests and generates validation report.
"""
import sys
import subprocess
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_pytest(test_path: str, markers: str = "") -> Tuple[bool, str, Dict[str, Any]]:
    """
    Run pytest and capture results.
    
    Returns:
        (passed, output, stats)
    """
    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
        "-v",
        "--tb=short",
        "-q"
    ]
    
    if markers:
        cmd.extend(["-m", markers])
    
    print(f"\n🧪 Running: pytest {test_path}")
    
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    
    output = result.stdout + result.stderr
    passed = result.returncode == 0
    
    # Parse stats from output
    stats = {"passed": 0, "failed": 0, "skipped": 0}
    for line in output.split("\n"):
        if "passed" in line or "failed" in line:
            # Try to extract numbers
            import re
            matches = re.findall(r"(\d+) (passed|failed|skipped)", line)
            for count, status in matches:
                stats[status] = int(count)
    
    return passed, output, stats


def check_system_components() -> Dict[str, Dict[str, Any]]:
    """Check all system components."""
    print("\n🔍 Checking system components...")
    
    results = {}
    
    # Check Qdrant
    try:
        from backend.core.qdrant_store import get_qdrant_connector
        connector = get_qdrant_connector()
        connected = connector.check_connection()
        collections = connector.list_collections() if connected else []
        results["qdrant"] = {
            "status": "connected" if connected else "error",
            "collections": collections
        }
        print(f"   Qdrant: {'✅' if connected else '❌'} {len(collections)} collections")
    except Exception as e:
        results["qdrant"] = {"status": "error", "message": str(e)}
        print(f"   Qdrant: ❌ {e}")
    
    # Check Ollama
    try:
        from backend.core.llm_client import get_llm_client
        client = get_llm_client()
        available = client.check_available()
        results["ollama"] = {
            "status": "available" if available else "unavailable",
            "model": client.model
        }
        print(f"   Ollama: {'✅' if available else '⚠️'} {client.model}")
    except Exception as e:
        results["ollama"] = {"status": "error", "message": str(e)}
        print(f"   Ollama: ❌ {e}")
    
    # Check Embedding
    try:
        from backend.core.embeddings import get_embedding_model
        model = get_embedding_model()
        info = model.get_info()
        results["embedding"] = {
            "status": "ready",
            "model": info["model_name"],
            "device": info["device"]
        }
        print(f"   Embedding: ✅ {info['device']}")
    except Exception as e:
        results["embedding"] = {"status": "error", "message": str(e)}
        print(f"   Embedding: ❌ {e}")
    
    # Check GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
        results["gpu"] = {
            "available": gpu_available,
            "name": gpu_name
        }
        print(f"   GPU: {'✅' if gpu_available else '⚠️'} {gpu_name or 'Not available'}")
    except Exception as e:
        results["gpu"] = {"available": False, "error": str(e)}
        print(f"   GPU: ⚠️ {e}")
    
    return results


def generate_report(
    system_check: Dict[str, Any],
    unit_tests: Tuple[bool, str, Dict],
    integration_tests: Tuple[bool, str, Dict],
    stability_result: Dict[str, Any] = None
) -> str:
    """Generate markdown validation report."""
    
    now = datetime.now()
    
    # Determine overall status
    all_passed = (
        unit_tests[0] and 
        integration_tests[0] and
        system_check.get("qdrant", {}).get("status") == "connected"
    )
    
    if stability_result:
        all_passed = all_passed and stability_result.get("status") == "STABLE"
    
    status_emoji = "✅" if all_passed else "❌"
    status_text = "PASS" if all_passed else "FAIL"
    
    report = f"""# Backend Validation Report

**Generated:** {now.strftime("%Y-%m-%d %H:%M:%S")}  
**Status:** {status_emoji} **{status_text}**

---

## System Components

| Component | Status | Details |
|-----------|--------|---------|
| Qdrant | {system_check.get('qdrant', {}).get('status', 'unknown')} | {len(system_check.get('qdrant', {}).get('collections', []))} collections |
| Ollama | {system_check.get('ollama', {}).get('status', 'unknown')} | {system_check.get('ollama', {}).get('model', 'N/A')} |
| Embedding | {system_check.get('embedding', {}).get('status', 'unknown')} | {system_check.get('embedding', {}).get('device', 'N/A')} |
| GPU | {'Available' if system_check.get('gpu', {}).get('available') else 'Not available'} | {system_check.get('gpu', {}).get('name', 'N/A')} |

---

## Test Results

### Unit Tests

| Metric | Value |
|--------|-------|
| Status | {'✅ PASS' if unit_tests[0] else '❌ FAIL'} |
| Passed | {unit_tests[2].get('passed', 0)} |
| Failed | {unit_tests[2].get('failed', 0)} |
| Skipped | {unit_tests[2].get('skipped', 0)} |

### Integration Tests

| Metric | Value |
|--------|-------|
| Status | {'✅ PASS' if integration_tests[0] else '❌ FAIL'} |
| Passed | {integration_tests[2].get('passed', 0)} |
| Failed | {integration_tests[2].get('failed', 0)} |
| Skipped | {integration_tests[2].get('skipped', 0)} |

"""
    
    # Add stability results if available
    if stability_result:
        report += """---

## Stability Test

| Endpoint | Requests | Success Rate | Avg Latency | P95 Latency |
|----------|----------|--------------|-------------|-------------|
"""
        for name, test in stability_result.get("tests", {}).items():
            success_rate = 100 - test.get("error_rate", 0)
            report += f"| {name} | {test.get('total_requests', 0)} | {success_rate:.1f}% | {test.get('avg_latency_ms', 0):.0f}ms | {test.get('p95_latency_ms', 0):.0f}ms |\n"
        
        report += f"""
**Stability Status:** {stability_result.get('status', 'UNKNOWN')}

"""
    
    report += """---

## Verification Commands

```powershell
# Run unit tests only
pytest tests/unit/ -v -m "unit"

# Run integration tests
pytest tests/integration/ -v -m "integration"

# Run stability check
python tests/stability_check.py --url http://127.0.0.1:8080

# Run all tests
pytest tests/ -v
```

---

*Report generated by `scripts/run_validation.py`*
"""
    
    return report


async def main():
    """Run full validation suite."""
    print("=" * 60)
    print("🔬 BACKEND VALIDATION SUITE")
    print(f"   Time: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # 1. System component check
    system_check = check_system_components()
    
    # 2. Unit tests
    print("\n" + "=" * 60)
    unit_passed, unit_output, unit_stats = run_pytest(
        "tests/unit/",
        markers="unit and not slow"
    )
    print(f"   Result: {'✅ PASS' if unit_passed else '❌ FAIL'}")
    print(f"   Stats: {unit_stats}")
    
    # 3. Integration tests (skip if Qdrant not connected)
    print("\n" + "=" * 60)
    if system_check.get("qdrant", {}).get("status") == "connected":
        int_passed, int_output, int_stats = run_pytest(
            "tests/integration/",
            markers="integration"
        )
        print(f"   Result: {'✅ PASS' if int_passed else '❌ FAIL'}")
        print(f"   Stats: {int_stats}")
    else:
        print("⚠️ Skipping integration tests - Qdrant not connected")
        int_passed, int_output, int_stats = False, "Skipped", {"passed": 0, "failed": 0, "skipped": 0}
    
    # 4. Stability check (optional - requires running server)
    stability_result = None
    try:
        import httpx
        response = httpx.get("http://127.0.0.1:8080/health", timeout=2)
        if response.status_code == 200:
            print("\n" + "=" * 60)
            from tests.stability_check import run_stability_check
            stability_result = await run_stability_check()
    except Exception as e:
        print(f"\n⚠️ Skipping stability check - Server not running: {e}")
    
    # 5. Generate report
    print("\n" + "=" * 60)
    print("📝 Generating validation report...")
    
    report = generate_report(
        system_check,
        (unit_passed, unit_output, unit_stats),
        (int_passed, int_output, int_stats),
        stability_result
    )
    
    # Save report
    report_path = PROJECT_ROOT / "BACKEND_VALIDATION_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"✅ Report saved to: {report_path}")
    
    # Overall result
    all_passed = unit_passed and (int_passed or system_check.get("qdrant", {}).get("status") != "connected")
    
    print("\n" + "=" * 60)
    print(f"{'✅ VALIDATION PASSED' if all_passed else '❌ VALIDATION FAILED'}")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
