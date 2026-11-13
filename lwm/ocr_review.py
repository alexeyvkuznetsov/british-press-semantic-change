"""Диагностика OCR: орфографический словарь не является фильтром.
Решения пользователя сохраняются отдельно от результатов диагностики.
"""
from pathlib import Path
import pandas as pd

DECISION_COLUMNS = ['neighbor', 'decision', 'decision_reason']
RULE_COLUMNS = ['period', 'target', 'neighbor', 'reason']


def load_decisions(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=DECISION_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    missing = set(DECISION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'{path.name}: отсутствуют столбцы {sorted(missing)}')
    df = df[DECISION_COLUMNS].copy()
    df['decision'] = df['decision'].str.strip().str.lower()
    if df['neighbor'].eq('').any() or df['neighbor'].duplicated().any():
        raise ValueError(f'{path.name}: пустое или повторяющееся слово neighbor')
    if not df['decision'].isin(['', 'keep', 'exclude', 'review']).all():
        raise ValueError(f'{path.name}: допустимы keep, exclude, review или пустая строка')
    if (df['decision'].isin(['keep', 'exclude']) & df['decision_reason'].str.strip().eq('')).any():
        raise ValueError(f'{path.name}: для keep/exclude укажите decision_reason')
    return df


def effective_rules(base_rules, decisions):
    """Старые точные правила сохраняются, подтверждённые exclude добавляются глобально."""
    extra = decisions.loc[decisions['decision'].eq('exclude'), ['neighbor','decision_reason']].copy()
    extra = extra.rename(columns={'decision_reason':'reason'}).assign(period='*', target='*')
    return pd.concat([base_rules[RULE_COLUMNS], extra[RULE_COLUMNS]], ignore_index=True).drop_duplicates(
        ['period','target','neighbor'], keep='first').reset_index(drop=True)


def diagnose(review, decisions, spell, *, topn=30, suggest=True, dictionary_version='', distance=1):
    """Одна строка на уникальную форму. Исправления — только подсказки для текущего Top-N."""
    r = review.copy()
    r['in_raw_top30'] = r['raw_rank'].le(topn)
    r['in_current_top30'] = ~r['excluded'] & r['filtered_rank'].le(topn)
    stats = r.groupby('neighbor', sort=True).agg(
        occurrences=('neighbor','size'), best_raw_rank=('raw_rank','min'),
        max_cosine=('cosine','max'), raw_top30_hits=('in_raw_top30','sum'),
        current_top30_hits=('in_current_top30','sum'), excluded_occurrences=('excluded','sum'),
        targets=('target', lambda x: '; '.join(sorted(set(x)))),
        periods=('period', lambda x: '; '.join(sorted(set(x))))
    ).reset_index()
    stats = stats.merge(decisions, on='neighbor', how='outer', validate='one_to_one')
    for c in ['decision','decision_reason','targets','periods']:
        stats[c] = stats[c].fillna('')
    for c in ['occurrences','raw_top30_hits','current_top30_hits','excluded_occurrences']:
        stats[c] = stats[c].fillna(0).astype(int)
    words = stats.loc[stats['occurrences'].gt(0),'neighbor'].tolist()
    unknown = spell.unknown(words)
    stats['spellchecker_unknown'] = stats['neighbor'].map(lambda w: w in unknown)
    stats['in_current_data'] = stats['occurrences'].gt(0)
    stats['dictionary_version'] = dictionary_version
    stats['suggestion_distance'] = distance
    stats['suggested_correction'] = ''
    pending = stats['decision'].isin(['','review'])
    active = stats['current_top30_hits'].gt(0)
    # Не используем unknown как правило исключения. Не предлагаем исправления
    # для каждой из тысяч форм: только для актуальных кандидатов без решения.
    if suggest:
        for i, word in stats.loc[active & pending & stats['spellchecker_unknown'],'neighbor'].items():
            correction = spell.correction(word)
            if correction and correction != word:
                stats.at[i,'suggested_correction'] = correction
    stats['priority'] = 'later'
    stats.loc[active & pending, 'priority'] = 'check_top30'
    stats.loc[active & pending & stats['spellchecker_unknown'], 'priority'] = 'unknown_top30'
    stats.loc[~pending, 'priority'] = 'decided'
    stats.loc[~stats['in_current_data'], 'priority'] = 'not_in_current_data'
    stats['keep_rule_conflict'] = stats['decision'].eq('keep') & stats['excluded_occurrences'].gt(0)
    order = {'unknown_top30':0,'check_top30':1,'later':2,'decided':3,'not_in_current_data':4}
    stats['_order'] = stats['priority'].map(order)
    stats = stats.sort_values(['_order','current_top30_hits','best_raw_rank','occurrences','neighbor'],
                              ascending=[True,False,True,False,True], na_position='last').drop(columns='_order')
    cols = ['neighbor','decision','decision_reason','priority','spellchecker_unknown','suggested_correction',
            'current_top30_hits','raw_top30_hits','best_raw_rank','occurrences','excluded_occurrences',
            'max_cosine','targets','periods','keep_rule_conflict','in_current_data','dictionary_version','suggestion_distance']
    return stats[cols].reset_index(drop=True)


def save_review(table, path):
    """Атомарная запись: существующие ручные решения уже включены в table."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    table.to_csv(tmp, index=False, encoding='utf-8-sig')
    tmp.replace(path)
