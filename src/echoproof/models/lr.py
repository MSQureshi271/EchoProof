"""Disposable LR fixture model with ordered class-1 chunk scores."""
from types import MappingProxyType
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from echoproof.contracts import ChunkScore, FeatureBatch, validate_chunk_scores
from echoproof.features.acoustic import FEATURE_NAMES, FEATURE_SCHEMA_HASH, handcrafted


class FixtureLRAdapter:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.metadata = MappingProxyType({"scope": "software_probe", "feature_schema_hash": FEATURE_SCHEMA_HASH})

    @classmethod
    def fit(cls, waveforms, labels, seed=42):
        x = np.stack([handcrafted(w) for w in waveforms])
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                             ("scaler", StandardScaler()), ("classifier", LogisticRegression(random_state=seed, max_iter=500))])
        pipeline.fit(x, np.asarray(labels))
        if list(pipeline.named_steps["classifier"].classes_) != [0, 1]:
            raise ValueError("fixture training requires both binary classes")
        return cls(pipeline)

    def prepare_features(self, chunks):
        """chunks contain (chunk_id, recording_id, valid_mono_waveform)."""
        x = np.stack([handcrafted(item[2]) for item in chunks])
        x = self.pipeline.named_steps["imputer"].transform(x)
        batch = FeatureBatch(x, [item[0] for item in chunks], [item[1] for item in chunks], None, None, None, FEATURE_SCHEMA_HASH)
        batch.validate()
        return batch

    def score_chunks(self, batch):
        batch.validate()
        if batch.feature_schema_hash != FEATURE_SCHEMA_HASH or batch.values.shape[1] != len(FEATURE_NAMES):
            raise ValueError("incompatible feature schema")
        classes = self.pipeline.named_steps["classifier"].classes_
        index = list(classes).index(1)
        probabilities = self.pipeline.predict_proba(batch.values)[:, index]
        scores = [ChunkScore(chunk_id, float(score)) for chunk_id, score in zip(batch.chunk_ids, probabilities)]
        validate_chunk_scores(batch.chunk_ids, scores)
        return scores

    def save(self, destination):
        joblib.dump({"scope": "software_probe", "feature_names": FEATURE_NAMES,
                     "feature_schema_hash": FEATURE_SCHEMA_HASH, "pipeline": self.pipeline}, destination)

    @classmethod
    def load(cls, source):
        # joblib may execute code; only load files produced by this trusted probe.
        data = joblib.load(source)
        if data.get("scope") != "software_probe" or tuple(data.get("feature_names", ())) != FEATURE_NAMES or data.get("feature_schema_hash") != FEATURE_SCHEMA_HASH:
            raise ValueError("incompatible fixture model")
        return cls(data["pipeline"])
