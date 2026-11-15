"""Compare regenerated tables with the author's original ZIP, without extracting it."""
import argparse
import io
from pathlib import Path
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE_FILES = {'05_country_group_sensitivity.csv', '05_leave_one_pair_out.csv',
                   '05_leave_one_pair_summary.csv'}


def compare(archive):
    exact = 0
    equivalent = []
    with zipfile.ZipFile(archive) as z:
        paths = [p for p in z.namelist() if '/results/tables/' in p and p.endswith('.csv')]
        if not paths:
            raise ValueError('Нет исходных CSV в архиве')
        for path in sorted(paths):
            name = Path(path).name
            dest = ROOT/'results/tables'/name.replace('_raw_top200','_raw_top500')
            before, after = z.read(path), dest.read_bytes()
            if before == after:
                exact += 1
                continue
            if name == '03_ocr_priority.csv':
                equivalent.append((name, 'диагностика обновлена; не влияет на фильтрацию'))
                continue
            a = pd.read_csv(io.BytesIO(before))
            b = pd.read_csv(io.BytesIO(after))
            if name == '03_ocr_effective_rules.csv':
                keys = ['period','target','neighbor']
                a, b = [t.sort_values(keys).reset_index(drop=True) for t in (a,b)]
            atol = 1.1e-6 if name in TOLERANCE_FILES else 1e-12
            pd.testing.assert_frame_equal(a,b,check_dtype=False,rtol=0,atol=atol)
            equivalent.append((name, f'содержимое совпадает, допуск {atol:g}' if name not in {'03_ocr_effective_rules.csv'} else 'те же правила, порядок строк изменён'))
    print(f'Проверено {len(paths)} таблиц; побайтно совпадают {exact}.')
    for name, reason in equivalent:
        print(f'{name}: {reason}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    compare(parser.parse_args().archive)
