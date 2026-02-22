# NewsletterAgent

Daily legal/tech intelligence pipeline with two output modes:

- **Digest mode (default):** multi-section newsletter
- **Single feature mode:** one synthesis-heavy strategic intelligence article/day

## Configuration

Edit `config/settings.yaml`:

```yaml
pipeline:
  single_feature_mode: false
  single_feature_top_k: 8
  single_feature_min_cluster_size: 3
```

### Single Feature Mode Behavior

When `single_feature_mode: true`, the pipeline runs a 4-stage generation architecture:

1. **Signal Clustering**
   - Inputs top-K scored items
   - Groups by thematic overlap
   - Selects strongest cluster (must meet `single_feature_min_cluster_size`)
2. **Thesis Extraction**
   - Produces one declarative structural-shift thesis for law firm partner/operators
3. **Full Synthesis Draft**
   - Writes one article using:
     - Trigger
     - Translation
     - Revenue Implication
     - Tactical Moves (3–5)
     - Risk/Blindspot
     - Strategic Takeaway
   - Uses sources inline as evidence (not source-by-source summaries)
4. **SEO Layer**
   - Declarative H1
   - H2 flow: What happened → Why it matters → What to do
   - Primary keyword in first 100 words
   - Numbered tactical subheads
   - Internal-link hooks (placeholders allowed)
   - Meta description + target keyword
   - Forward-looking close

## Policy Gates (Single Feature)

Single-feature output is blocked unless all pass:

- cluster meets minimum size
- at least 2 independent sources
- at least 2 source domains

## Running

```bash
python -m scripts.run_pipeline
```

## Examples

### Keep default digest flow

```yaml
pipeline:
  single_feature_mode: false
```

### Enable single synthesis feature

```yaml
pipeline:
  single_feature_mode: true
  single_feature_top_k: 8
  single_feature_min_cluster_size: 3
```
