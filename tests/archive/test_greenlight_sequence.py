import csv
import pytest
from slowlab.archive.greenlight_sequence import concatenate_greenlight_inputs


def write_day(path, description="value"):
    with path.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["Time","x"]);w.writerow(["time",description]);w.writerow(["s","-"])
        w.writerow([0,1]);w.writerow([43200,2]);w.writerow([86400,3])


def test_concatenation_keeps_one_boundary_and_monotone_absolute_time(tmp_path):
    a=tmp_path/"a.csv";b=tmp_path/"b.csv";out=tmp_path/"out.csv";write_day(a);write_day(b)
    assert concatenate_greenlight_inputs([a,b],out)==5
    with out.open() as f:r=csv.DictReader(f);next(r);next(r);rows=list(r)
    assert [float(x["Time"]) for x in rows]==[0,43200,86400,129600,172800]


def test_concatenation_rejects_metadata_drift(tmp_path):
    a=tmp_path/"a.csv";b=tmp_path/"b.csv";write_day(a);write_day(b,"changed")
    with pytest.raises(ValueError,match="metadata changed"):
        concatenate_greenlight_inputs([a,b],tmp_path/"out.csv")
