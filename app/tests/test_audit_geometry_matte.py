"""Audit 02/03 regressions on actual crop, matte and photometric routines."""
import threading
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pytest
from roop.alignment_safety import InvalidAlignment, TrackGeometry, affine, unit_latent
from roop.face_util import estimate_norm, swap_template_points
from roop.face_analyser import canonicalize_face_alignment
from roop.unified_matte import inward_alpha, warp_premultiplied, yaw_from_kps
from roop.procmgr_masking import MaskingMixin
from roop.procmgr_color import ColorTransferMixin
from roop.procmgr_merger import MergerMixin
from roop.orientation import resolve_track_rolls

P = swap_template_points(256, 'arcface').astype(np.float32)
BOX = np.array([0,0,256,256],np.float32)

@pytest.mark.parametrize('bad', [np.zeros((5,2)), np.full((5,2),np.nan), P[[1,0,2,4,3]], P*0.00001])
def test_bad_fits_are_rejected(bad):
    with pytest.raises(InvalidAlignment):
        canonicalize_face_alignment(np.zeros((256,256,3),np.uint8),
            dict(kps=bad,bbox=BOX),256,'arcface')

@pytest.mark.parametrize('roll', [0,40,55,90,135,180,270])
def test_roll_is_valid_at_every_orientation(roll):
    theta=np.radians(roll)
    r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    pts=(P-128)@r.T+128
    m=estimate_norm(pts,256)
    assert np.linalg.det(m[:,:2])>0
    np.testing.assert_allclose(pts@m[:,:2].T+m[:,2],P,atol=1e-4)


def test_bbox_and_reflection_are_rejected():
    with pytest.raises(InvalidAlignment):
        canonicalize_face_alignment(np.zeros((256,256,3),np.uint8),dict(kps=P+900,bbox=BOX),256,'arcface')
    with pytest.raises(InvalidAlignment): affine(np.array([[-1,0,0],[0,1,0]]))


def test_temporal_ema_resets_and_does_not_alias():
    g=TrackGeometry()
    g.update(1,0,P,BOX)
    q=g.update(1,1,P+1,BOX)
    np.testing.assert_allclose(q,P+.65,atol=1e-5)
    q[:]=0
    np.testing.assert_allclose(g.update(1,2,P+2,BOX,cut=True),P+2)
    np.testing.assert_allclose(g.update(1,3,P+100,BOX),P+100)


def test_track_scale_guard():
    g=TrackGeometry(); g.update(1,0,P,BOX)
    with pytest.raises(InvalidAlignment): g.update(1,1,(P-128)*.3+128,BOX)


def test_parallel_track_state_is_isolated():
    g=TrackGeometry()
    def run(tid):
        for i in range(20): q=g.update(tid,i,P+i*.1,BOX)
        return q
    with ThreadPoolExecutor(max_workers=4) as e:
        result=list(e.map(run,range(4)))
    for q in result: np.testing.assert_array_equal(q,result[0])

@pytest.mark.parametrize('v',[np.zeros(512,np.float16), np.full(512,np.nan), np.full(512,np.inf)])
def test_invalid_identity_latent_fails_clearly(v):
    with pytest.raises(ValueError): unit_latent(v)


def test_fp16_norm_is_promoted():
    out=unit_latent(np.full(512,2000,np.float16))
    assert out.dtype==np.float32
    assert np.linalg.norm(out)==pytest.approx(1.,abs=1e-6)


def test_first_rotation_needs_continuity():
    pts=P.copy()
    pts[:,1]=256-pts[:,1]; pts[:,0]=256-pts[:,0]
    faces=[dict(kps=pts.copy()) for _ in range(3)]
    resolve_track_rolls(faces)
    assert faces[0]['rotation_action'] is None
    assert faces[1]['rotation_action'] is None
    assert faces[2]['rotation_action']=='rotate_180'


def test_inward_feather_and_yaw_only_remove_alpha():
    mask=np.zeros((128,128),np.float32); mask[12:116,20:108]=1
    front=inward_alpha(mask,10)
    for yaw in [-60,60]:
        profile=inward_alpha(mask,10,yaw)
        assert np.all(profile[mask==0]==0)
    assert np.all(front[mask==0]==0)
    assert front[64,21]<front[64,28]<front[64,64]
    assert front[64,64]==1


def test_premultiplied_edges_do_not_admit_background():
    crop=np.full((64,64,3),255,np.uint8); crop[12:52,12:52]=(20,60,100)
    a=np.zeros((64,64),np.float32); a[12:52,12:52]=1
    im=np.array([[1.,0,.35],[0,1.,.6]])
    colour,wa=warp_premultiplied(crop,a,im,(80,80))
    np.testing.assert_allclose(colour[wa>0],np.tile([20,60,100],(np.count_nonzero(wa),1)),atol=1e-4)
    assert np.all(wa[64:]==0)


class Paste(MaskingMixin):
    def __init__(self):
        self.options=SimpleNamespace(blend_ratio=1.,show_face_area_overlay=False)
        self._tls=threading.local()

@pytest.mark.parametrize('offset',[-30,0,70])
def test_paste_preserves_plate_outside_engine_support(offset,monkeypatch):
    monkeypatch.setenv('ROOP_GRAIN_MATCH','0')
    crop=np.full((64,64,3),(40,60,220),np.uint8)
    plate=np.full((128,128,3),90,np.uint8)
    m=np.array([[1.,0.,-offset],[0.,1.,-offset]])
    alpha=np.zeros((64,64),np.float32); alpha[16:48,16:48]=1
    out=Paste().paste_upscale(crop,crop,m,plate,1.,[0,0,0,0,12.,10.],crop_alpha=alpha)
    footprint=cv2.warpAffine(alpha,cv2.invertAffineTransform(m),(128,128))
    np.testing.assert_array_equal(out[footprint==0],plate[footprint==0])
    assert np.any(out!=plate)

@pytest.mark.parametrize('mode',['lct','rct','mkl','idt'])
def test_colour_statistics_ignore_crop_background(mode):
    rng=np.random.default_rng(19)
    source=rng.integers(60,180,(64,64,3),dtype=np.uint8)
    target=rng.integers(60,180,(64,64,3),dtype=np.uint8)
    mask=np.zeros((64,64),bool); mask[16:48,16:48]=True
    a=getattr(ColorTransferMixin(), '_color_transfer_'+mode)(source,target,mask=mask)
    source[~mask]=255; target[~mask]=0
    b=getattr(ColorTransferMixin(), '_color_transfer_'+mode)(source,target,mask=mask)
    np.testing.assert_array_equal(a[mask],b[mask])


def test_histogram_statistics_ignore_background():
    rng=np.random.default_rng(30)
    source=rng.integers(60,180,(64,64,3),dtype=np.uint8)
    target=rng.integers(60,180,(64,64,3),dtype=np.uint8)
    mask=np.zeros((64,64),np.float32); mask[16:48,16:48]=1
    m=MergerMixin(); a=m.apply_hist_match(source,target,1,mask=mask)
    source[mask==0]=255; target[mask==0]=0
    b=m.apply_hist_match(source,target,1,mask=mask)
    np.testing.assert_array_equal(a[mask>0],b[mask>0])


def test_affine_scale_guard_runs_before_warp_and_supports_sar():
    g=TrackGeometry()
    a=np.array([[1.4,0,0],[0,1.,0]])
    g.validate_matrix(1,0,a,sar=1.4)
    with pytest.raises(InvalidAlignment): g.validate_matrix(1,1,a*3,sar=1.4)
    # A real shot change accepts a new scale without blending histories.
    g.validate_matrix(1,1,a*3,sar=1.4,cut=True)


def test_replayed_frame_never_advances_track_state():
    g=TrackGeometry(); g.update(1,10,P,BOX)
    g.update(1,9,P+20,BOX)
    np.testing.assert_allclose(g.update(1,11,P+1,BOX),P+.65,atol=1e-5)


def test_frame_and_track_roll_share_the_same_68_axis():
    from roop.orientation import roll_from_face
    from roop.face_util import face_roll_tilt
    from insightface.app.common import Face
    lm=np.zeros((68,3),np.float32); lm[27,:2]=[128,100]; lm[8,:2]=[150,200]
    f=Face(kps=P.copy(),landmark_3d_68=lm)
    assert face_roll_tilt(f)==pytest.approx(roll_from_face(f))
    assert abs(face_roll_tilt(f))>10


def load_tests(loader, tests, pattern):
    from tests.unittest_shim import load_tests_for
    return load_tests_for(globals())


def test_yaw_proxy_is_roll_invariant_and_changes_sign():
    pts=P.copy(); pts[2,0]+=25
    yaw=yaw_from_kps(pts)
    assert yaw>10
    r=np.array([[0.,-1.],[1.,0.]],np.float32)
    assert yaw_from_kps(pts@r.T)==pytest.approx(yaw,abs=1e-4)
    pts[2,0]-=50
    assert yaw_from_kps(pts)<-10


@pytest.mark.parametrize('index',[0,5])
def test_writer_keeps_and_checks_explicit_frame_binding(index,monkeypatch):
    from queue import Queue
    import roop.globals
    from roop.ProcessMgr import ProcessMgr
    m=ProcessMgr.__new__(ProcessMgr)
    m.num_threads=1; m.processed_queue=[Queue()]
    frame=np.full((8,8,3),(13,70,211),np.uint8)
    m.processed_queue[0].put((True,(index,frame)))
    m.processed_queue[0].put((False,None))
    encoded=[]
    m.output_to_file=True; m.output_to_cam=False
    m.videowriter=SimpleNamespace(write_frame=encoded.append)
    m._temporal_faces=None
    m._runtime_adaptive_boundary=lambda: None
    monkeypatch.setattr(roop.globals,'processing',True)
    m.write_frames_thread()
    if index==0:
        assert len(encoded)==1
        np.testing.assert_array_equal(encoded[0],frame)
        assert getattr(m,'_writer_error',None) is None
    else:
        assert not encoded
        assert isinstance(m._writer_error,RuntimeError)


@pytest.mark.parametrize('matrix',[np.zeros((2,3)),np.full((2,3),np.nan),np.array([[-1.,0,0],[0,1.,0]])])
def test_invalid_paste_matrix_retains_plate(matrix):
    crop=np.full((64,64,3),200,np.uint8); plate=np.zeros((128,128,3),np.uint8)
    out=Paste().paste_upscale(crop,crop,matrix,plate,1.,[0,0,0,0,12.,10.])
    np.testing.assert_array_equal(out,plate)



def test_matrix_guard_rejects_before_sampling_crop():
    dst=np.full((256,256,3),177,np.uint8)
    def reject(matrix): raise InvalidAlignment('rejected by track')
    with pytest.raises(InvalidAlignment):
        canonicalize_face_alignment(np.zeros_like(dst),dict(kps=P,bbox=BOX),256,'arcface',
                                    dst=dst,matrix_guard=reject)
    assert np.all(dst==177)



def test_mask_temporal_histories_are_isolated_per_engine():
    from roop.one_euro import MaskStabilizer
    st=MaskStabilizer()
    zero=np.zeros((64,64),np.float32); one=np.ones_like(zero)
    for t in range(4):
        np.testing.assert_array_equal(st.apply(zero,P,t,track_id=1,stream_key='xseg'),zero)
        np.testing.assert_array_equal(st.apply(one,P,t,track_id=1,stream_key='occluder'),one)
    assert len(st.tracks)==2
