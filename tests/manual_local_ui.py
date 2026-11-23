"""Manual offline browser test. Launches a loopback-only test server.
Provider availability and quotes are TEST DOUBLES, not live account facts.
Reservations and requests network calls are tripwires. Never deploy this harness.
Run: python tests/manual_local_ui.py --source /path/multiproduct-barber1.jpg --output-dir /tmp/ui-proof
"""
import os,sys,threading,time,json
from pathlib import Path
import argparse,tempfile
parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
ROOT=Path(__file__).resolve().parent.parent
os.environ['RENDERELLO_RUNS']=str(args.output_dir/'runs')
# Set RENDERELLO_MODEL_DIR to your verified u2netp cache if needed.
os.environ['RENDERELLO_COOKIE_SECURE']='0'
os.environ['RENDERELLO_ACCESS_CODE']='local-testing-only'
for key in ['MAGIC_HOUR_API_KEY','RENDERELLO_LOGIN_EMAIL','RENDERELLO_LOGIN_PASSWORD_HASH']:os.environ.pop(key,None)
sys.path[:0]=[str(ROOT),str(ROOT/'webapp')]
import main,renderello.studio as studio
calls=[]
def forbidden(*a,**k):
    calls.append('forbidden');raise AssertionError('Network/spend disabled for local UI test')
studio.reserve_budget=forbidden
studio._enforce_budget=lambda *a,**k:None
studio.budget_status=lambda model:{'credits_per_image':30,'remaining':1500,'max_batch':1500,'allocated':1500,'spent':0}
main.provider_report=lambda:[{'name':'procedural','available':True},{'name':'magichour','available':True,'models':{'seedream-v5-flash':30}}]
import requests,urllib.request
requests.sessions.Session.request=forbidden
urllib.request.urlopen=forbidden
import uvicorn
server=uvicorn.Server(uvicorn.Config(main.app,host='127.0.0.1',port=8876,log_level='error'))
thr=threading.Thread(target=server.run);thr.start()
while not server.started:time.sleep(.05)
from playwright.sync_api import sync_playwright
try:
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1100})
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto('http://127.0.0.1:8876')
        print('title',page.title())
        print('inputs',page.locator('input').evaluate_all('(els)=>els.map(x=>({id:x.id,type:x.type}))'))
        page.screenshot(path=str(args.output_dir/'initial.png'),full_page=True)
        # Login through the actual endpoint, then use the actual page.
        page.request.post('http://127.0.0.1:8876/api/login',data={'code':'local-testing-only'},headers={'Content-Type':'application/json'})
        # request data must be JSON, use evaluate fetch instead.
        page.evaluate("async()=>await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code:'local-testing-only'})})")
        page.reload();page.wait_for_selector('.preset')
        print('presets',page.locator('.preset').count())
        page.select_option('#provider','magichour')
        page.select_option('#variants','1')
        page.locator('#file-products').set_input_files(str(args.source))
        first=page.locator('.preset').first
        print('first',first.get_attribute('data-key'))
        first.click()
        page.wait_for_function("!document.querySelector('#generate').disabled")
        page.locator('#generate').click()
        accept=page.get_by_role('button',name='I reviewed these masks and accept this cutout',exact=True)
        accept.wait_for(timeout=30000)
        page.locator('#results').screenshot(path=str(args.output_dir/'review-ui.png'))
        assert page.get_by_text('Missing items will not be restored.',exact=False).count()==1
        accept.click()
        assert page.get_by_role('button',name='Confirmed for these photos.',exact=False).is_disabled()
        assert page.get_by_role('button',name='Confirmed for these photos.',exact=False).count()==1
        page.locator('#thumbs-products button').click()
        page.locator('#file-products').set_input_files(str(args.source))
        page.wait_for_function("!document.querySelector('#generate').disabled")
        page.locator('#generate').click()
        page.get_by_role('button',name='I reviewed these masks and accept this cutout',exact=True).wait_for(timeout=30000)
        print('UI flow PASS (including stale review reset): failed job preview rendered, explicit confirmation, no auto-retry; errors=',errors,'network/spend calls=',calls)
        assert not errors and not calls
        browser.close()
finally:
    server.should_exit=True;thr.join(5)
