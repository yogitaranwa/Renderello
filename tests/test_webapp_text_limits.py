"""Metadata boundary checks only. Never executes a generation worker."""
import importlib
import io
import secrets
import sys
from pathlib import Path
import pytest
from PIL import Image
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'webapp')]
CASES=[('ascii_2000','x'*2000,200),('ascii_2001','x'*2001,400),('emoji_2000','🙂'*2000,200),('emoji_2001','🙂'*2001,400),('crlf_2000','x\r\n'*999+'xx',200),('crlf_2001','x\r\n'*1000+'x',400)]
@pytest.mark.parametrize('field',['instruction','product_description'])
@pytest.mark.parametrize('label,text,status',CASES,ids=[c[0]for c in CASES])
def test_text_limit_matches_unicode_and_multiline_form(monkeypatch,tmp_path,field,label,text,status):
 code=secrets.token_urlsafe(24)
 monkeypatch.setenv('RENDERELLO_ACCESS_CODE',code);monkeypatch.setenv('RENDERELLO_COOKIE_SECURE','0');monkeypatch.setenv('RENDERELLO_RUNS',str(tmp_path/'runs'))
 for key in ['RENDERELLO_LOGIN_EMAIL','RENDERELLO_LOGIN_PASSWORD_HASH']:monkeypatch.delenv(key,raising=False)
 import main,auth
 importlib.reload(main);auth._attempts.clear()
 called=[]
 async def metadata_only(*args,**kwargs):called.append(args);return {'job_id':'metadata-only-no-generation'}
 monkeypatch.setattr(main,'_generate_claimed',metadata_only)
 from fastapi.testclient import TestClient
 with TestClient(main.app)as client:
  assert client.post('/api/login',json={'code':code}).status_code==200
  buf=io.BytesIO();Image.new('RGB',(16,16),'blue').save(buf,'PNG')
  response=client.post('/api/generate',files={'products':('source.png',buf.getvalue(),'image/png')},data={'preset':'clean_studio','provider':'procedural','variants':'1',field:text})
  assert response.status_code==status,(label,response.text)
  if status==200:
   actual=called[0][6 if field=='instruction'else 7]
   assert actual==text.replace('\r\n','\n').replace('\r','\n')
   assert len(actual)==2000
  else:assert not called and not main._jobs
 auth._attempts.clear()
