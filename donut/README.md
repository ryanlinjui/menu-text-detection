### Menu Image Type
- h: horizontal
- v: vertical
- d: document
- s: scene
- i: irregular

https://github.com/clovaai/donut
https://huggingface.co/spaces/naver-clova-ix/donut-base-finetuned-cord-v2/blob/main/requirements.txt
.venv/lib/python3.10/site-packages/donut/model.py - 443
    - if self.device.type == "cuda" or "mps": 
.venv/lib/python3.10/site-packages/donut/model.py - 453
    - if self.device.type != "cuda" and self.device.type != "mps":
.venv/lib/python3.10/site-packages/donut/util.py
    - ground_truth = json.loads(json.dumps(eval(sample["ground_truth"])))

multiprocessing_context='fork'

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

```
{
  "restaurant": "",
  "address": "",
  "phone": "",
  "business_hours": "",
  "items": [
    {
      "name": "",
      "price": ""
    },
    {
      "name": "",
      "price": ""
    }
  ]
},

---

value 允許為空字串，你要判斷什麼時候是空字串，price 可能是數字或是時價，再下一次的每次的問答我會給你text sequence，請依照以上的json格式只產生給我json就好
```    