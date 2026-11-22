"""Run the actual component JS against a small canvas/message fixture."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from agentrestoration.core import ROOT


def test_canvas_drag_coordinates_clear_and_disabled(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for component JS testing")
    html = (ROOT/"ui/components/region_selector/index.html").read_text(encoding="utf-8")
    javascript = re.search(r"<script>(.*?)</script>",html,re.S).group(1)
    fixture = r'''
const assert=require("node:assert/strict"),vm=require("node:vm");
const messages=[],handlers={},elements={};
const context={clearRect(){},drawImage(){},fillRect(){},strokeRect(){}};
for(const id of ["canvas","clear","status"]){
 elements[id]={style:{},addEventListener:(name,fn)=>handlers[id+":"+name]=fn};
}
Object.assign(elements.canvas,{getContext:()=>context,setPointerCapture(){},clientWidth:400,clientHeight:200,
 getBoundingClientRect:()=>({left:100,top:50,width:400,height:200})});
const parent={postMessage:m=>messages.push(m)};
const sandbox={console,document:{getElementById:id=>elements[id],body:{getBoundingClientRect:()=>({height:260})}},
 window:{parent,addEventListener:(name,fn)=>handlers[name]=fn},
 Image:class{constructor(){this.naturalWidth=800;this.naturalHeight=400;} set src(value){this.url=value;this.onload();} get src(){return this.url;}},
 ResizeObserver:class{constructor(fn){this.fn=fn;} observe(){this.fn();}}};
vm.runInNewContext(COMPONENT_SOURCE,sandbox);
function render(disabled=false,region=null){
 handlers.message({source:parent,data:{type:"streamlit:render",args:{image:"data:image/png;base64,fixture",image_id:"one",disabled,region}}});
}
function event(clientX,clientY){return {clientX,clientY,pointerId:1,preventDefault(){}};}
function values(){return messages.filter(m=>m.type==="streamlit:setComponentValue");}
render();
assert(messages.some(m=>m.type==="streamlit:componentReady"&&m.apiVersion===1));
handlers["canvas:pointerdown"](event(400,200));
handlers["canvas:pointermove"](event(200,100));
handlers["canvas:pointerup"](event(200,100));
let result=values().at(-1);
assert.equal(result.value.image_id,"one");
assert.deepEqual(JSON.parse(JSON.stringify(result.value.region)),{x0:.25,y0:.25,x1:.75,y1:.75});
handlers["clear:click"]();assert.equal(values().at(-1).value.region,null);
render(true);let count=values().length;
handlers["canvas:pointerdown"](event(200,100));handlers["canvas:pointerup"](event(400,200));
assert.equal(values().length,count);
render();handlers["canvas:pointerdown"](event(0,0));handlers["canvas:pointerup"](event(999,999));
assert.deepEqual(JSON.parse(JSON.stringify(values().at(-1).value.region)),{x0:0,y0:0,x1:1,y1:1});
render();handlers["canvas:pointerdown"](event(201,101));handlers["canvas:pointerup"](event(201.5,101.5));
assert.equal(values().at(-1).value.region,null);
'''
    import json
    script=tmp_path/"canvas.cjs"
    script.write_text(fixture.replace("COMPONENT_SOURCE",json.dumps(javascript)),encoding="utf-8")
    result=subprocess.run([node,str(script)],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
