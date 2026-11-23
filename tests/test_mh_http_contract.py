"""No network. Fake HTTP contract tests against the owning API documentation.
https://docs.magichour.ai/api-reference/image-projects/ai-image-editor.md
https://docs.magichour.ai/api-reference/files/generate-asset-upload-urls.md
https://docs.magichour.ai/api-reference/image-projects/get-image-details.md
"""
import io
import json
import pytest
import requests
from PIL import Image
from renderello.backends.magichour import MagicHourBackend, MagicHourError
from renderello.schemas import Aspect

class Reply:
    status_code = 200
    def __init__(self, data=None, content=b''):
        self.data, self.content = data, content
    def json(self):
        return self.data
    def raise_for_status(self):
        pass


def test_submission_receipt_then_poll_reference_and_charge(monkeypatch, tmp_path):
    calls = []
    buf = io.BytesIO(); Image.new('RGB',(16,16),'orange').save(buf,'PNG')
    receipt = tmp_path/'receipts.jsonl'
    statuses = iter(['queued','rendering','complete'])
    def post(url, **kw):
        calls.append(('POST',url,kw))
        if url.endswith('upload-urls'):
            assert kw['json'] == {'items':[{'type':'image','extension':'png'}]}
            return Reply({'items':[{'upload_url':'https://upload.invalid/a','file_path':'a.png'}]})
        payload = kw['json']
        assert payload['resolution'] == '640px'
        assert payload['assets']['image_file_paths'] == ['a.png','a.png']
        assert 'seed' not in payload and 'negative_prompt' not in payload
        return Reply({'id':'project-1','credits_charged':10})
    def get(url, **kw):
        calls.append(('GET',url,kw))
        assert receipt.exists(), 'ID must be persisted before polling'
        if url.endswith('image-projects/project-1'):
            st=next(statuses)
            return Reply({'status':st,'credits_charged':9,
                          'downloads':[{'url':'https://output.invalid/a.png'}] if st=='complete' else []})
        return Reply(content=buf.getvalue())
    monkeypatch.setenv('MAGIC_HOUR_API_KEY','test-only-not-a-real-key')
    monkeypatch.setattr(requests,'post',post)
    monkeypatch.setattr(requests,'put',lambda *a,**k:Reply())
    monkeypatch.setattr(requests,'get',get)
    b=MagicHourBackend(live=True,receipt_path=receipt,poll_interval=0,cost_log=tmp_path/'cost.jsonl')
    result=b.edit(Image.new('RGB',(16,16)), 'scene', 'neg', 0, Aspect.SQUARE,
                  reference=Image.new('RGB',(16,16)))
    assert result.size==(16,16)
    assert json.loads(receipt.read_text())['project_id']=='project-1'
    assert json.loads((tmp_path/'cost.jsonl').read_text())['credits_charged']==9
    assert sum(c[0]=='POST' and c[1].endswith('ai-image-editor') for c in calls)==1


def test_poll_timeout_and_resume_never_posts(monkeypatch,tmp_path):
    monkeypatch.setenv('MAGIC_HOUR_API_KEY','test-only')
    monkeypatch.setattr(requests,'post',lambda *a,**k:pytest.fail('no generation on recovery'))
    monkeypatch.setattr(requests,'get',lambda *a,**k:Reply({'status':'rendering'}))
    b=MagicHourBackend(receipt_path=tmp_path/'receipt',poll_timeout=0)
    with pytest.raises(MagicHourError,match='resume this ID'):
        b.resume('existing-id')


def test_missing_live_charge_stays_unknown(tmp_path):
    b=MagicHourBackend(live=True,cost_log=tmp_path/'cost.jsonl')
    b._log_cost(0,Aspect.SQUARE,1,{'status':'complete'})
    assert json.loads((tmp_path/'cost.jsonl').read_text())['credits_charged'] is None
