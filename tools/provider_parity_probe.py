"""Small provider-change arm used by the actual acceptance report."""
from __future__ import annotations
import base64, json, re, sys, time
from pathlib import Path
import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "acceptance_monica_harjot"
BASE = "http://127.0.0.1:17862"
TARGET = Path(r"D:\Monica Bellucci .mp4")
SOURCE = ROOT / "app" / "facesets" / "harjot.fsz"
LOG = OUT / "server_cpu.log"

def req(method, path, **kw):
    kw.setdefault("timeout", 180)
    r = requests.request(method, BASE + path, **kw)
    try: b = r.json()
    except Exception: b = {"text": r.text[:2000]}
    return r.status_code, b

def dec(data):
    if not data or "," not in data: return None
    return cv2.imdecode(np.frombuffer(base64.b64decode(data.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)

def routes(text):
    out=[]
    for line in text.splitlines():
        if "[SelectedRoute]" not in line: continue
        m=re.search(r"frame=(\d+).*?swapped=\[(.*?)\]", line)
        if not m: continue
        out.append({"frame":int(m.group(1)), "swapped":[] if not m.group(2).strip() else [int(x) for x in re.findall(r"\d+",m.group(2))], "line":line.strip()})
    return out

def wait_ready():
    for _ in range(180):
        try:
            c,b=req("GET","/api/meta",timeout=5)
            if c==200 and b.get("active_provider"): return b
        except Exception: pass
        time.sleep(1)
    raise RuntimeError("CPU provider API did not become ready")

def payload(settings, frame, fake):
    return {"index":0,"frame":frame,"fake_preview":fake,"detection":"Selected face",
            "selection_state":{"selection_mode":"selected","person_id":0},"face_mapping":[0],"source_index":0,
            "swap_model":settings.get("swap_model","hyperswap"),"enhancer":settings.get("selected_enhancer","None"),
            "mask_engine":settings.get("mask_engine","None"),"face_distance":settings.get("max_face_distance",1.2),
            "blend_ratio":settings.get("blend_ratio",1.0),"autorotate":settings.get("autorotate_faces",True),
            "refine_landmarks":settings.get("refine_landmarks",True),"use_source_bank":settings.get("use_source_bank",True),
            "use_3d_recon":settings.get("use_3d_recon",True),"upscale_after_swap":False,
            "processing_method":settings.get("video_swapping_method","In-Memory processing")}

def main():
    report=json.loads((OUT/"acceptance_report.json").read_text(encoding="utf-8"))
    primary=report["preflight"]["active_runtime"]
    meta=wait_ready()
    req("POST","/api/source/clear"); req("POST","/api/target/clear")
    with SOURCE.open("rb") as fh:
        sc,sb=req("POST","/api/source/add",files={"files":(SOURCE.name,fh,"application/octet-stream")})
    ac,ab=req("POST","/api/target/add_path",json={"paths":[str(TARGET)]})
    target_media_id = ab.get("target_media_id") if isinstance(ab, dict) else None
    if not target_media_id:
        raise RuntimeError("target add response did not provide target_media_id")
    cc,cb=req("POST","/api/target/use_face",json={"index":0,"frame":61,
                                                       "target_media_id":target_media_id,
                                                       "face_index":0})
    settings=req("GET","/api/settings")[1]
    rc,raw=req("POST","/api/preview",json=payload(settings,1,False))
    fc,fake=req("POST","/api/preview",json=payload(settings,1,True))
    cpu_ids=fake.get("person_ids",[]) if isinstance(fake,dict) else []
    primary_d=report["stages"]["D_preview"]
    primary_ids=primary_d["selected"].get("person_ids",[])
    # Run two frames so this arm also emits selected-route evidence.
    req("POST","/api/target/set_frame",json={"which":"start","frame":0})
    req("POST","/api/target/set_frame",json={"which":"end","frame":2})
    start=LOG.stat().st_size if LOG.exists() else 0
    sw,body=req("POST","/api/swap",json={**payload(settings,1,False),"target_index":0,"end_frame":2})
    deadline=time.time()+900; last={}
    while time.time()<deadline:
        _,last=req("GET","/api/progress",timeout=30)
        if not last.get("processing"): break
        time.sleep(2)
    text=LOG.read_text(encoding="utf-8",errors="replace") if LOG.exists() else ""
    cpu_routes=[x for x in routes(text[start:]) if x.get("frame") in (0,1)]
    result={"provider":meta.get("active_provider"),"requested_provider":meta.get("requested_provider"),
            "admitted_provider":meta.get("admitted_provider"),"source_http":sc,"target_http":ac,"capture_http":cc,
            "source_response":sb,"target_response":ab,"capture_response":cb,"raw_http":rc,"preview_http":fc,
            "cpu_preview_person_ids":cpu_ids,"primary_preview_person_ids":primary_ids,
            "preview_person_id_mapping_equal":cpu_ids==primary_ids,
            "cpu_routes":cpu_routes,"cpu_render_http":sw,"cpu_progress":last,
            "selection_state":{"selection_mode":"selected","person_id":0},
            "target_person_id":0,"chosen_source_index":0,"threshold":settings.get("max_face_distance"),
            "runtime_lines":[x.strip() for x in text[start:].splitlines() if "[Runtime]" in x],
            "primary_provider":primary.get("active_provider"),
            "errors":last.get("error") if isinstance(last,dict) else None}
    # The product's route lines are the final authority for the rendered arm.
    result["cpu_selected_route_present"]=bool(cpu_routes)
    result["pass"]=bool(meta.get("active_provider")=="cpu" and result["preview_person_id_mapping_equal"] and result["cpu_selected_route_present"] and not result["errors"])
    (OUT/"provider_parity.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    report["provider_parity"]=result
    report["acceptance"]["7_provider_change_does_not_change_routing"]=bool(result["pass"])
    if all(v is True for k,v in report["acceptance"].items()):
        report["status"]="PASS_WITH_INPUT_NAME_CORRECTION"
    else:
        report["status"]="FAIL"
    (OUT/"acceptance_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    stages = report["stages"]
    lines = [
        "# Roop Ultimate acceptance test: Harjot → Monica Bellucci", "",
        f"Status: **{report['status']}**", "",
        f"Requested target: `{report['requested']['target_literal']}` — literal exists: `{report['discovery']['literal_exists']}`.",
        f"Tested discovered target: `{report['discovery']['selected_path']}`.",
        f"Faceset: `{report['discovery']['source_path']}`; reference faces/angles: `{report['faceset']['reference_face_count']}` / `{report['faceset']['reference_angles']}`.",
        f"Primary active provider: `{primary.get('active_provider')}`; swap model: `{report['model']['swap_model']}`.",
        "", "| Stage | Result | Evidence |", "|---|---|---|",
    ]
    for name, value in stages.items():
        if isinstance(value, dict):
            routes_count = len(value.get("routes", [])) if isinstance(value.get("routes"), list) else "n/a"
            lines.append(f"| {name} | {'PASS' if value.get('pass') else 'FAIL/SKIPPED'} | routes={routes_count}, output={value.get('output_path', '')} |")
    lines += ["", "Provider parity:", f"- CPU active provider: `{result['provider']}`; TensorRT active provider: `{result['primary_provider']}`.",
              f"- Preview person-ID mapping equal: `{result['preview_person_id_mapping_equal']}`; CPU selected-route render present: `{result['cpu_selected_route_present']}`.", "", "Acceptance criteria:"]
    for key, value in report["acceptance"].items(): lines.append(f"- {key}: `{value}`")
    lines += ["", "Machine-readable report: `acceptance_report.json`."]
    (OUT/"acceptance_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"pass":result["pass"],"provider":result["provider"],"cpu_routes":len(cpu_routes),"mapping_equal":result["preview_person_id_mapping_equal"]},indent=2))
    return 0 if result["pass"] else 1

if __name__=="__main__": raise SystemExit(main())
