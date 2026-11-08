# Benchmark baseline và agent

## Luồng chạy

- Baseline: retrieval một lần → chọn context → generate.
- Agent: retrieval gốc và tối đa một query expansion → verify evidence → generate;
  dùng Tavily để phục hồi khi evidence thiếu, generator từ chối hoặc source ID lỗi.
- Hai luồng dùng chung corpus, hybrid retriever, context selector, prompt và generator.
- Reranker tắt; candidate pool 30 mỗi nhánh, RRF k=60, retrieval top-k=10.
- Tối đa 30 evidence mỗi lần generate; sibling enrichment vẫn được giữ.
- Generator gọi model một lần mỗi `generate()`. Không có lượt sửa citation ẩn,
  tự điền doc_id, suy đoán nguồn từ locator hoặc kiểm tra chuỗi pháp lý trong runtime.
- Runtime chỉ resolve source ID và render URL web từ metadata. Độ đúng của nội dung,
  locator và dẫn chiếu được đo bởi bộ chấm.
- Agent mặc định có 8 lượt planner, 4 lượt component và 1 lượt web. Query expansion
  và generation là các lượt LLM riêng; budget component không phải budget LLM.

`GENERATOR_MODEL` phải được đặt rõ trong môi trường hoặc `--model`;
`GENERATOR_PROVIDER`/`--provider` chọn `google` hoặc `openai`. Judge chọn nhà
cung cấp bằng `EVALUATOR_PROVIDER`/`--provider`, model bằng `EVALUATOR_MODEL`/
`--model`. Google dùng `GOOGLE_API_KEY`/`GEMINI_API_KEY`; OpenAI dùng
`OPENAI_API_KEY`. Model judge và generator độc lập.

GPT-6 Luna dùng ID `gpt-6-luna`. Generator dùng Responses API, mặc định reasoning
`none` để giữ budget sinh ngắn; judge OpenAI mặc định reasoning `medium`, dùng
Responses structured outputs với schema Pydantic. `--reasoning-effort` chọn
`none`, `low`, `medium`, `high`, `xhigh`, `max`. Không gửi temperature cho các
request này. Budget output gồm cả reasoning; tăng `--max-tokens` ở generator
hoặc `--max-output-tokens` ở judge khi cần.

Từ thư mục gốc project trên Colab, sau khi đặt key trong môi trường hoặc `.env`:

```python
!python -m test.evaluation_agent.collect --split pilot --provider openai --model gpt-6-luna --reasoning-effort low --max-tokens 4096 --output Data/test/results/pilot_outputs_luna.jsonl
!python -m test.evaluation_agent.score --split pilot --provider openai --model gpt-6-luna --reasoning-effort medium --max-output-tokens 8192 --input Data/test/results/pilot_outputs_luna.jsonl --output Data/test/results/pilot_scores_luna.jsonl --include-details
```

Chỉ muốn chấm lại output cũ thì truyền đường dẫn output v2 vào `--input` và
ghi score sang file Luna mới. Giữ cấu hình provider/model/reasoning/budget khi
resume. Đổi provider/model không được dùng chung checkpoint. SDK OpenAI được
ghi phiên bản trong metadata, không dùng lại điểm Gemini ngầm định.

Tham khảo: [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna),
[structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

## Cài đặt

Dùng các dependency trong `requirements.txt` ở thư mục gốc:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip check
```

Ragas 0.3.1 và các phiên bản SDK/LangChain được pin để giữ tương thích. Không
nâng riêng dependency giữa một lần benchmark đang resume.

## Chạy một câu

```powershell
python -m test.evaluation_agent --show-config
python -m test.evaluation_agent baseline "Câu hỏi" --query-id q_001 --output Data/test/results/outputs_new.jsonl --include-trace
python -m test.evaluation_agent agent "Câu hỏi" --query-id q_001 --output Data/test/results/outputs_new.jsonl --include-trace
```

Khi dùng `query_id` trong dataset, truyền nguyên văn câu hỏi của ID đó. Runner
chỉ nhận câu hỏi, không nhận reference, category hoặc nguồn gold. Chạy tuần tự
vì mỗi runner giữ diagnostics của lần gọi gần nhất.

Output gồm `query_id`, `variant`, `query`, `answer`, `contexts`, `sources`,
`status`, `latency_ms`, `error`. Context/source gồm toàn bộ input của lần sinh
answer cuối, theo cùng renderer và thứ tự S1, S2…; không chỉ những nguồn đã trích.
Trace thêm `generation_attempts` và `execution_trace`, giữ cả lần bị từ chối.
Thông báo từ chối do graph tạo không bị gán context của bản nháp đã loại.

Status có thể là `answered`, `refused`, `citation_invalid`, `budget_exhausted`
hoặc `error`. Một lỗi công cụ đã được phục hồi vẫn có thể nằm trong `error`;
đọc `status` để xác định kết quả cuối.

## Thu output và chấm

Sau đợt dọn code, dùng tên file mới cho pipeline và scorer. Ví dụ từ thư mục
project trên Colab, thêm `!` trước `python`:

```bash
python -m test.evaluation_agent.collect --split pilot --output Data/test/results/pilot_outputs_clean.jsonl
python -m test.evaluation_agent.score --input Data/test/results/pilot_outputs_clean.jsonl --split pilot --check-only
python -m test.evaluation_agent.score --input Data/test/results/pilot_outputs_clean.jsonl --output Data/test/results/pilot_scores_clean.jsonl --split pilot --model YOUR_JUDGE_MODEL --include-details
```

Pilot gồm 10 câu × 2 luồng. Collector luân phiên thứ tự baseline/agent, lưu
trace và cấu hình. Sau khi duyệt pilot, dùng `--split test` và tên file test mới.
Đo latency chính thức cần warm-up riêng; collector không tự gọi thêm dịch vụ
để warm-up. Chi phí khởi tạo model/kết nối nằm ngoài timer `run()`.

Bộ chấm đọc output đã lưu, không chạy lại retrieval/generation. Có thể chấm lại
output v2 bằng scorer mới, nhưng phải ghi sang file điểm mới; để đo pipeline
đã dọn thì cần thu lại output cho cả hai luồng.

## Checkpoint và resume

Thêm `--resume` vào đúng lệnh đã chạy khi bị gián đoạn:

- Collector lưu cấu hình và `pipeline_sha256` trong `.run.json`; bỏ qua cặp
  ID/variant đã lưu, kể cả lỗi, không chọn lần sinh tốt nhất.
- Scorer lưu checkpoint ngay trong score `.jsonl`, cùng `.meta.json` chứa
  model, cấu hình, dependency và hash code/dataset/output.
- Scorer giữ metric `ok`, `na`, `input_error`; thử lại `judge_error` và chấm
  các metric còn thiếu. `--include-details` checkpoint cả các bước trace Ragas.
- Resume yêu cầu metadata khớp hoàn toàn. Thay model, code hoặc dữ liệu phải
  dùng file điểm mới. Giữ nguyên các file v5; không sửa metadata để vượt kiểm tra.
- Chỉ một tiến trình ghi mỗi file. File `.tmp` còn sót khi tiến trình bị ngắt
  giữa lúc ghi cần được kiểm tra thủ công trước khi tiếp tục.

Lỗi HTTP 400/401/403/404/429 được lưu rồi dừng batch. Retry request tạm thời
được giới hạn ba lần, timeout 90 giây. Với 429, dùng hướng dẫn của server hoặc
backoff 10/20 giây, tối đa 60 giây; quota yêu cầu chờ lâu hơn thì dừng. SDK và
Ragas không tự thêm lớp retry khác. Thông báo lỗi được che khóa/token/URL.

`--allow-partial` cho phép kiểm tra file chưa đủ hai luồng, không biến kết quả
thiếu thành benchmark hoàn chỉnh. Các CLI chẩn đoán riêng từng câu và migration
điểm từ schema cũ đã được bỏ; `--include-details` và `--resume` vẫn được giữ.

## Ba metric

- **Factual Correctness:** Ragas F1, atomicity/coverage cao. Answer và reference
  được chuẩn bị cùng quy tắc: bỏ annotation citation khi không hỏi căn cứ;
  giữ số liệu, điều kiện, phủ định, tham chiếu thực chất và căn cứ được hỏi.
  Câu hỏi chỉ xác định phạm vi, không trở thành bằng chứng.
- **Faithfulness:** Ragas supported/total, chỉ dùng context thực tế của answer
  cuối. Judge chọn evidence ID trong catalog; code resolve nguyên văn/offset
  và kiểm tra địa chỉ/quan hệ pháp lý. Không lấy gold hoặc context của bản nháp.
  Kiểm tra cấu trúc không thay thế suy luận ngữ nghĩa của judge.
- **Citation F1:** claim gắn với citation trong cùng câu/block. Judge chỉ được
  xem những nguồn được trích cho claim đó. Precision đếm liên kết claim–nguồn
  đóng góp, recall đếm claim được hỗ trợ đầy đủ, loại trùng claim/source.
  Chuỗi dẫn chiếu cần đủ nguồn; nguồn dư làm giảm precision. Thiếu nội dung
  reference ảnh hưởng Factual Correctness, không tự thêm vào citation recall.

Không có metric tổng hợp. Không claim/citation thì `null`; có claim nhưng
không citation thì Citation F1=0. Có claim mà không context thì Faithfulness=0.
Câu answerable bị từ chối/rỗng thì Factual Correctness=0. Với nhóm
`insufficient_evidence`, factual không áp dụng và hành vi cần duyệt theo rubric.
Lỗi API/schema/input không được tự quy thành 0 hoặc 1.

`--include-details` giữ claim/verdict/lý do, TP/FP/FN, evidence resolve và citation
P/R để kiểm tra lại. Trace quan sát các lượt chấm hiện có, không gọi thêm judge
để giải thích hoặc chọn điểm tốt hơn. Statement NLI sai thứ tự/thiếu/thừa, ID giả
hoặc quan hệ cấu trúc sai được ghi `judge_error`. Mọi trace chứa answer/reference
nên được bảo vệ như dữ liệu benchmark.

### Ước lượng token nếu dùng GPT làm judge

Đếm cục bộ bằng `tiktoken`, dùng lại prompt và luồng gọi của cả ba metric:

```powershell
.\venv\Scripts\python.exe -m test.evaluation_agent.estimate_tokens --input Data/test/results/test_outputs_pipeline_v2.jsonl --model gpt-5 --output Data/test/results/test_tokens_gpt5.json
```

Mặc định chọn split `test`; dùng `--split pilot` hoặc `--allow-partial` khi cần.
`--scores Data/test/results/test_scores_pipeline_v5.jsonl` dùng lại analysis và
claim decomposition đã lưu cho những dòng có trace. Dataset/output hash và
prompt version phải khớp; mọi dòng vẫn được tính như một lần chấm mới đầy đủ.
Các dòng thiếu trace ước lượng một claim mỗi block không phải tiêu đề, nên số
claim và nhánh từ chối có thể khác judge thật.

Báo cáo JSON tách token prompt, system instruction, JSON schema và overhead API
giả định (`--framing-tokens`, mặc định 12/lượt), theo metric, variant và từng
output/lượt gọi. Bước analysis dùng chung chỉ tính một lần. Các kịch bản
`--output-tokens-per-call 512 1024 2048` là ngân sách token sinh **gồm reasoning**,
không phải dự đoán độ dài output hoặc giới hạn bảo đảm. Không tính retry, cache,
retriever/generator; không gọi API và không ghi điểm metric giả.

Lần đầu `tiktoken` cần tải vocabulary công khai. Có thể đặt `TIKTOKEN_CACHE_DIR`
đến thư mục cache có quyền ghi; khi cache đã có thì phần đếm không cần mạng.
`--model` của lệnh ước lượng chỉ chọn tokenizer, không chọn provider chấm.
Model không có mapping sẽ báo lỗi; chỉ dùng `--encoding` khi biết encoding phù hợp.

## Kiểm tra offline

Các unit test của bộ benchmark nằm trong `test/unit/`: runner, lưu kết quả,
dataset, chấm ba metric và phạm vi nội dung, địa chỉ bằng chứng, dẫn chiếu pháp
lý, OpenAI và bảo vệ checkpoint khi resume. Các test generator/agent đã có
trong Git vẫn ở thư mục gốc `test/`.

```powershell
.\venv\Scripts\python.exe -m unittest discover -s test -t .
```

Test dùng component/judge giả. Chúng kiểm tra luồng chạy, checkpoint, schema,
phép tính và lỗi cấu trúc; không chứng minh model thật luôn trả lời/chấm đúng.
