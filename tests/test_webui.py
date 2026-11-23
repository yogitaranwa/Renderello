import io

from renderello.webui import _parse_multipart, serve


def _multipart(fields, boundary="XBOUND"):
    body = b""
    for name, (fname, data, ctype) in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="{name}"' + (f'; filename="{fname}"' if fname else "")
                 + f"\r\nContent-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body


def test_multipart_parser_roundtrip():
    body = _multipart({
        "template": ("", b"clean_studio", "text/plain"),
        "photo": ("p.png", b"\x89PNG-fake-bytes", "image/png"),
    })
    fields = _parse_multipart(body, 'multipart/form-data; boundary=XBOUND')
    assert fields["template"] == ("", b"clean_studio")
    assert fields["photo"] == ("p.png", b"\x89PNG-fake-bytes")


def test_webui_end_to_end(tmp_path, monkeypatch):
    import sys
    import urllib.request
    from tests.conftest import make_product
    # Force the grabcut fallback path deterministically: this test covers the
    # GrabCut review warning, not backend auto-selection (which depends on
    # whether torch/onnxruntime happen to be installed locally).
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    server = serve(0, tmp_path)  # port 0 = ephemeral
    port = server.server_address[1]
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "Renderello" in page and "clean_studio" in page
        # PDF section 4 flow coverage: visual template preview + aspect control.
        assert '/preview/clean_studio.png' in page
        assert 'name="aspect"' in page and '4:5' in page

        buf = io.BytesIO()
        make_product().save(buf, "PNG")
        body = _multipart({
            "photo": ("product.png", buf.getvalue(), "image/png"),
            "template": ("", b"clean_studio", "text/plain"),
            "instruction": ("", b"softer light", "text/plain"),
            "aspect": ("", b"4:5", "text/plain"),
            "variants": ("", b"2", "text/plain"),
            "route": ("", b"cutout_compose", "text/plain"),
        })
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/run", data=body,
            headers={"Content-Type": "multipart/form-data; boundary=XBOUND"})
        res = urllib.request.urlopen(req).read().decode()
        assert "Results" in res
        assert res.count("cutout_compose seed") == 2
        assert "auto fidelity" in res
        assert "REVIEW REQUIRED: GrabCut fallback" in res
        # Download and edit/regenerate controls (PDF section 4).
        assert res.count("Download PNG") == 2
        assert 'action="/rerun"' in res

        # Edit/regenerate reuses the stored source with fresh seeds.
        import re as _re
        job = _re.search(r'name="job" value="([a-f0-9]+)"', res).group(1)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/rerun",
            data=f"job={job}&template=luxury_hero&aspect=1%3A1&variants=2&route=cutout_compose".encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        res2 = urllib.request.urlopen(req).read().decode()
        assert res2.count("cutout_compose seed") == 2
        assert "luxury_hero" in res2

        # Path traversal is refused.
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/file/..%2F..%2Fetc%2Fpasswd")
            raise AssertionError("traversal should fail")
        except urllib.error.HTTPError as e:
            assert e.code in (403, 404)
    finally:
        server.shutdown()


def test_scene_compose_option_present_and_mock_scene_shape():
    from renderello import webui
    import numpy as np
    g = np.zeros((40, 30, 3), np.uint8)
    out = webui._mock_scene_fn(g, "x")
    assert out.shape == (40, 30, 3) and out.dtype == np.uint8


def test_webui_scene_compose_route_runs_with_mock_scene(tmp_path):
    import threading, urllib.request
    from tests.conftest import make_product
    server = serve(0, tmp_path); port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert 'value="scene_compose"' in page and "MOCK" in page
        buf = io.BytesIO(); make_product().save(buf, "PNG")
        body = _multipart({
            "photo": ("product.png", buf.getvalue(), "image/png"),
            "template": ("", b"luxury_hero", "text/plain"),
            "instruction": ("", b"", "text/plain"),
            "aspect": ("", b"4:5", "text/plain"),
            "variants": ("", b"2", "text/plain"),
            "route": ("", b"scene_compose", "text/plain"),
        })
        req = urllib.request.Request(f"http://127.0.0.1:{port}/run", data=body,
                                     headers={"Content-Type": "multipart/form-data; boundary=XBOUND"})
        res = urllib.request.urlopen(req).read().decode()
        assert res.count("scene_compose seed") == 2 and "variant failed" not in res
    finally:
        server.shutdown()


def test_multipart_preserves_binary_trailing_newlines():
    data=b'\x89PNG-opaque\r\n\n\r'
    fields=_parse_multipart(_multipart({'photo':('p.png',data,'image/png')}),
                            'multipart/form-data; boundary=XBOUND')
    assert fields['photo'][1]==data


def test_reference_rerun_retains_backdrop_and_form_settings(tmp_path):
    import urllib.request, urllib.parse, re, json
    from PIL import Image
    from tests.conftest import make_product
    server=serve(0,tmp_path);url=f'http://127.0.0.1:{server.server_address[1]}'
    try:
        source=io.BytesIO();make_product().save(source,'PNG')
        ref=io.BytesIO();Image.new('RGB',(160,120),(80,170,110)).save(ref,'PNG')
        body=_multipart({'photo':('product.png',source.getvalue(),'image/png'),
             'reference':('ref.png',ref.getvalue(),'image/png'),
             'template':('',b'clean_studio','text/plain'),
             'variants':('',b'3','text/plain'),'route':('',b'scene_compose','text/plain')})
        req=urllib.request.Request(url+'/run',data=body,headers={'Content-Type':'multipart/form-data; boundary=XBOUND'})
        response=urllib.request.urlopen(req).read().decode()
        assert '<option value="scene_compose" selected>' in response
        assert '<option selected>3</option>' in response
        job=re.search(r'name="job" value="([a-f0-9]+)"',response).group(1)
        req=urllib.request.Request(url+'/rerun',data=urllib.parse.urlencode({
             'job':job,'template':'clean_studio','variants':3,'route':'scene_compose'}).encode())
        response2=urllib.request.urlopen(req).read().decode()
        newjob=re.search(r'name="job" value="([a-f0-9]+)"',response2).group(1)
        assert (tmp_path/newjob/'reference.png').read_bytes()==ref.getvalue()
        assert response2.count('scene_compose seed')==3
        for p in (tmp_path/newjob).rglob('scene_compose_seed*.png'):
            assert Image.open(p).getpixel((5,5))==(80,170,110)
    finally:
        server.shutdown()


def test_previews_ready_before_first_page(tmp_path):
    import urllib.request
    from renderello.templates import all_templates
    server=serve(0,tmp_path)
    try:
        for template in all_templates():
            response=urllib.request.urlopen(f'http://127.0.0.1:{server.server_address[1]}/preview/{template.key}.png')
            assert response.status==200 and response.read().startswith(b'\x89PNG')
    finally:
        server.shutdown()


def test_failed_web_job_has_no_download_or_quality_rating(tmp_path, monkeypatch):
    import urllib.request
    from PIL import Image
    # Fail every variant deterministically; the original flat image only failed
    # because of whichever cutout backend the environment resolved.
    from renderello import pipeline as pipeline_mod
    def _boom(*a, **k):
        raise RuntimeError("forced failure for failure-rendering test")
    # Fail the cutout step: the pipeline records failed variants instead of
    # raising, which is the rendering path under test.
    monkeypatch.setattr(pipeline_mod, "cutout", _boom)
    buf=io.BytesIO();Image.new('RGB',(300,300),(128,128,128)).save(buf,'PNG')
    server=serve(0,tmp_path);url=f'http://127.0.0.1:{server.server_address[1]}'
    try:
        body=_multipart({'photo':('flat.png',buf.getvalue(),'image/png'),'variants':('',b'2','text/plain'),'route':('',b'cutout_compose','text/plain')})
        req=urllib.request.Request(url+'/run',data=body,headers={'Content-Type':'multipart/form-data; boundary=XBOUND'})
        text=urllib.request.urlopen(req).read().decode()
        assert text.count('FAILED / NOT GENERATED')==2
        assert 'Download PNG' not in text and 'auto fidelity' not in text
        assert 'No image output and no quality rating' in text
        assert 'UNKNOWN / unclassified source' in text
        assert 'SHA256:' in text and 'No source or rights record' in text
    finally:server.shutdown()


def test_registered_photo_upload_exposes_content_source_provenance(tmp_path):
    import urllib.request
    from pathlib import Path
    src=Path(__file__).parents[1]/'benchmark/sources/watch1.jpg'
    server=serve(0,tmp_path);url=f'http://127.0.0.1:{server.server_address[1]}'
    try:
        body=_multipart({'photo':('renamed.jpg',src.read_bytes(),'image/jpeg'),'variants':('',b'2','text/plain'),'route':('',b'cutout_compose','text/plain')})
        req=urllib.request.Request(url+'/run',data=body,headers={'Content-Type':'multipart/form-data; boundary=XBOUND'})
        text=urllib.request.urlopen(req).read().decode()
        assert 'gadgetdude' in text and 'flickr.com/photos/' in text
        assert 'exact_file_sha256' in text and 'no current legal clearance' in text
    finally:server.shutdown()


def test_source_declarations_block_opaque_paste_and_survive_rerun(tmp_path):
    import urllib.request, urllib.parse, json, re
    from tests.conftest import make_product
    server=serve(0,tmp_path);url=f'http://127.0.0.1:{server.server_address[1]}'
    try:
        source=io.BytesIO();make_product().save(source,'PNG')
        fields={'photo':('p.png',source.getvalue(),'image/png'),
                'source_pose':('',b'front','text/plain'),
                'source_material':('',b'transparent','text/plain'),
                'source_framing':('',b'complete','text/plain'),
                'require_complete':('',b'yes','text/plain')}
        request=urllib.request.Request(url+'/run',data=_multipart(fields),headers={'Content-Type':'multipart/form-data; boundary=XBOUND'})
        text=urllib.request.urlopen(request).read().decode()
        assert 'Download PNG' not in text and text.count('FAILED / NOT GENERATED')==2
        assert 'Transparent source needs an explicit transmission workflow' in text
        assert 'value="transparent" selected' in text
        assert 'name="require_complete" value="yes" checked' in text
        job=re.search(r'name="job" value="([a-f0-9]+)"',text).group(1)
        request=urllib.request.Request(url+'/rerun',data=urllib.parse.urlencode({'job':job,'variants':2}).encode())
        text2=urllib.request.urlopen(request).read().decode()
        job2=re.search(r'name="job" value="([a-f0-9]+)"',text2).group(1)
        assert json.loads((tmp_path/job2/'source-review.json').read_text())==json.loads((tmp_path/job/'source-review.json').read_text())
        assert 'Download PNG' not in text2
        # Explicit unchecked new form may clear completeness without losing material.
        request=urllib.request.Request(url+'/rerun',data=urllib.parse.urlencode({'job':job2,'variants':2,'source_pose':'front'}).encode())
        text3=urllib.request.urlopen(request).read().decode()
        job3=re.search(r'name="job" value="([a-f0-9]+)"',text3).group(1)
        assert json.loads((tmp_path/job3/'source-review.json').read_text())['require_complete'] is False
    finally:server.shutdown()
