import numpy as np
import pandas as pd


def make_submission(valid_df, out_dir, output_path="submission.csv"):
    """Build submission CSV from saved series .npy files."""
    print("Making submission CSV ...")

    submit_df = []
    gb = valid_df.groupby("id")

    for i, (sample_id, df) in enumerate(gb):
        try:
            series = np.load(f"{out_dir}/digitalised/{sample_id}.series.npy")

            series_by_lead = {}
            for l in range(3):
                lead = [
                    ["I",   "aVR", "V1", "V4"],
                    ["II",  "aVL", "V2", "V5"],
                    ["III", "aVF", "V3", "V6"],
                ][l]

                length = [
                    df[df["lead"] == lead[j]].iloc[0].number_of_rows
                    for j in range(4)
                ]
                if lead[0] == "II":
                    length[0] = length[0] - sum(length[1:])

                index = np.cumsum(length)[:-1]
                split = np.split(series[l], index)
                for k, s in zip(lead, split):
                    series_by_lead[k] = s

            series_by_lead["II"] = series[3]

        except:
            series_by_lead = {}
            for j, d in df.iterrows():
                series_by_lead[d.lead] = np.zeros(d.number_of_rows)

        for j, d in df.iterrows():
            # Safety pad to correct length
            series_by_lead[d.lead] = np.concatenate([
                series_by_lead[d.lead],
                np.zeros_like(series_by_lead[d.lead]),
            ])[:d.number_of_rows]
            assert len(series_by_lead[d.lead]) == d.number_of_rows

            print(f"\r\t {i} {sample_id} : {d.lead}", end="", flush=True)

            row_id = [f"{sample_id}_{i}_{d.lead}" for i in range(d.number_of_rows)]
            this_df = pd.DataFrame({
                "id": row_id,
                "value": series_by_lead[d.lead].astype(np.float32),
            })
            submit_df.append(this_df)

    print("")
    submit_df = pd.concat(submit_df, axis=0, ignore_index=True, sort=False, copy=False)
    submit_df.to_csv(output_path, index=False)
    print(f"Saved {output_path} ({len(submit_df)} rows)")
    return submit_df