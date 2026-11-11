# Repo Analysis

**What are segments?**

A segment is one caption line exactly as YouTube returns it: a short piece of text plus the time it's shown on screen. It's stored as `CaptionSegment` in `backend/models/video.py`, with `text`, `start` (seconds) and `duration` (seconds). For example:

```
{"text": "so now let's create a variable", "start": 12.4, "duration": 2.1}
```

They're fetched in `backend/services/video_processor.py`, using `youtube-transcript-api`, and passed along unchanged. A video's transcript is a list of hundreds or thousands of these. Auto-generated captions usually have a few words per segment. Manually written captions can be longer, often a full sentence.

Chunking joins consecutive segments into bigger pieces. Because a chunk is made of whole segments, it knows exactly when it starts (its first segment's `start`) and ends (its last segment's `start + duration`). That's what lets citations link to the right moment in the video.

**What is the benefit of smaller vs larger chunk sizes?**

Smaller chunks (e.g. 200–500 characters):
- **More precise search.** Each chunk is about one idea, so its embedding closely matches questions about that idea.
- **More precise timestamps.** A citation points to a 15–30 second window instead of several minutes.
- **Cheaper prompts.** Sending `top_k` small chunks to the LLM uses fewer tokens.
- **Downside:** each chunk has little context. An answer that's spread over a longer explanation gets split across chunks, and some of them may not be retrieved. A sentence like "that's why we do it this way" can lose the context that explains what "it" means.

Larger chunks (e.g. 1000–2000 characters):
- **More context per chunk.** Whole explanations stay together, so the LLM sees the full reasoning.
- **Fewer chunks to store and embed.** Indexing is cheaper and faster.
- **Downside:** less precise search. A chunk covering several topics gets an embedding that blends them all, so it matches no single question strongly. Timestamps are also coarser, and each chunk sent to the LLM costs more tokens.

This repo uses 500 characters, which is on the small side. That favours precise search and timestamps over context.

**What is the benefit of smaller vs larger overlap?**

Overlap means the end of one chunk is repeated at the start of the next.

Larger overlap (e.g. 10–20% of the chunk size):
- **Less lost at boundaries.** A sentence or idea that falls across two chunks appears whole in at least one of them, so search can still find it.
- **Smoother context.** Each chunk starts with a little of what came before.
- **Downside:** more repeated text. That means more chunks, more embedding cost and more storage. Search results can also return near-identical neighbouring chunks, which wastes `top_k` slots.

Smaller or no overlap:
- **No repetition,** and the fewest chunks.
- **Downside:** ideas that fall on a boundary get cut in half, and neither half may match the question well.

This repo sets the overlap to 50 characters (10% of 500). But it only carries over whole segments that fit within 50 characters, so in practice the overlap is often just one short segment or nothing at all. A common default is around 10–20% of the chunk size, applied reliably.
