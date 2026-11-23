import pytest
from PIL import Image
from renderello.label_text_audit import parse_tsv,compare_words,extract_words
H='level\tleft\ttop\twidth\theight\tconf\ttext\n'
def word(t):return dict(token=t)
def test_tsv_filters_confidence_and_short_noise():
    t=H+'5\t1\t2\t30\t9\t90\tCoconut\n5\t1\t2\t3\t4\t12\tchanged\n5\t1\t2\t3\t4\t99\tof\n'
    assert parse_tsv(t)==[dict(token='COCONUT',observed_text='Coconut',confidence=90.,box=[1,2,30,9])]
def test_tsv_bad_geometry_and_header():
    assert parse_tsv(H+'5\t-1\t2\t4\t4\t90\tTEST\n')==[]
    with pytest.raises(ValueError):parse_tsv('junk')
def test_missing_print_warns():
    d=compare_words([word(x)for x in ['COCONUT','DIRECTIONS','INGREDIENTS']],[word('FAKETEXT')])
    assert d['source_word_recall']==0 and d['status']=='OCR_MISMATCH_REVIEW_REQUIRED'
    assert d['photo_accepted']is False and d['product_text_verified']is False
def test_full_match_not_acceptance():
    a=[word(x)for x in ['A','B','C']];d=compare_words(a,a)
    assert d['source_word_recall']==1 and d['status']=='NO_OCR_TRIPWIRE_NOT_A_TEXT_PASS'
    assert d['product_text_verified']is False
def test_empty_ocr_not_success():
    d=compare_words([],[]);assert d['source_word_recall']is None and d['status']=='INSUFFICIENT_SOURCE_OCR'
def test_duplicate_words_counted():
    d=compare_words([word('TEST')]*4,[word('TEST')]);assert d['source_word_recall']==.25
@pytest.mark.parametrize('rows',[[dict(token='lower')],['COCONUT'],None])
def test_untyped_records_rejected(rows):
    with pytest.raises(ValueError):compare_words(rows,[])
def test_missing_binary_explicit(monkeypatch):
    monkeypatch.setattr('renderello.label_text_audit.shutil.which',lambda x:None)
    with pytest.raises(RuntimeError,match='not installed'):extract_words(Image.new('RGB',(8,8)))
def test_mode_and_size_not_silently_changed():
    with pytest.raises(ValueError):extract_words(Image.new('L',(8,8)))
    with pytest.raises(ValueError):extract_words(Image.new('RGB',(2100,2100)))
def test_call_is_local_fixed_args_no_shell(monkeypatch):
    import subprocess
    monkeypatch.setattr('renderello.label_text_audit.shutil.which',lambda x:'/usr/bin/tesseract')
    def fake(args,**kw):
        assert args==['/usr/bin/tesseract','stdin','stdout','-l','eng','--psm','11','tsv']
        assert kw['timeout']==30 and 'shell'not in kw
        return subprocess.CompletedProcess(args,0,H.encode(),b'')
    monkeypatch.setattr('renderello.label_text_audit.subprocess.run',fake)
    assert extract_words(Image.new('RGB',(8,8)))==[]
