"""Asset-selection correction made before model loading or benchmark inference."""
import bootstrap as b
_original_get=b.json_get
_original_publish=b.publish
def get(url):
    obj=_original_get(url)
    if url.endswith('/releases/tags/'+b.TAG):
        obj['assets']=[a for a in obj['assets'] if a['name']==f'llama-{b.TAG}-bin-ubuntu-x64.tar.gz']
        if len(obj['assets'])!=1:raise ValueError('Expected exact official CPU asset')
    return obj
def publish(name,obj):
    return _original_publish('preflight-r2.json',obj)
b.json_get=get
if __name__=='__main__':
    b.publish=publish
    b.main()
