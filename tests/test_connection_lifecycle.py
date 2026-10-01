"""Connections must close deterministically, without relying on GC."""

import sqlite3
from contextlib import closing

import pytest
from test_dashboard_core import make_db

from dashboard_core import DashboardStore

# Cover each of the eleven read-only connection sites, including early returns.
READ_CASES = [
    pytest.param('realtime_event_snapshot', (), 1, id='realtime'),
    pytest.param('diagnostics', (), 1, id='diagnostics'),
    pytest.param('stats', (), 1, id='stats'),
    pytest.param('list_memories', (), 1, id='memories'),
    pytest.param('get_memory', ('w1',), 1, id='memory-found'),
    pytest.param('get_memory', ('missing',), 1, id='memory-missing'),
    pytest.param('triples', (), 1, id='triples'),
    pytest.param('consolidations', (), 1, id='consolidations'),
    # The first two connections are in list_memories() and consolidations().
    pytest.param('session_detail', ('s1',), 3, id='session-detail'),
    pytest.param('today_digest', ('2026-05-04',), 1, id='today'),
    pytest.param('memoria_stats', (), 1, id='memoria-stats'),
    pytest.param('memoria_facts', (), 1, id='memoria-table'),
]

WRITE_CASES = [
    pytest.param('invalidate_memory', (), 'status', 'expired', id='invalidate'),
    pytest.param('set_memory_importance', (0.2,), 'importance', 0.2, id='importance'),
    pytest.param('set_memory_veracity', ('inferred',), 'veracity', 'inferred', id='veracity'),
    pytest.param('set_memory_expiry', ('2020-01-01T00:00:00',), 'valid_until', '2020-01-01T00:00:00', id='expiry'),
    pytest.param('supersede_memory', ('Replacement content',), 'status', 'superseded', id='supersede'),
]


@pytest.fixture
def tracked_store(tmp_path, monkeypatch):
    db = tmp_path / 'mnemosyne.db'
    make_db(db)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'hermes'))
    opened = []
    real_connect = sqlite3.connect

    class TrackedConnection(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.events = []

        def __exit__(self, exc_type, exc_value, traceback):
            result = super().__exit__(exc_type, exc_value, traceback)
            self.events.append('rollback' if exc_type else 'commit')
            return result

        def close(self):
            super().close()
            self.events.append('close')

    def track_connect(*args, **kwargs):
        con = real_connect(*args, factory=TrackedConnection, **kwargs)
        # Keep strong references so automatic collection cannot hide a leak.
        opened.append(con)
        return con

    monkeypatch.setattr(sqlite3, 'connect', track_connect)
    try:
        yield DashboardStore(db), opened, real_connect
    finally:
        # Also clean up when an assertion exposes a leaking implementation.
        for con in opened:
            con.close()


def assert_closed(opened):
    assert opened, 'expected at least one database connection'
    for con in opened:
        assert con.events[-1:] == ['close']
        with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
            con.execute('SELECT 1')


@pytest.mark.parametrize('method,args,error_connection', READ_CASES)
@pytest.mark.parametrize('empty', [False, True], ids=['populated', 'empty'])
def test_read_connections_close(tracked_store, method, args, error_connection, empty):
    store, opened, real_connect = tracked_store
    with closing(real_connect(store.db_path)) as con, con:
        if empty:
            for table in ('working_memory', 'episodic_memory', 'triples', 'consolidation_log'):
                con.execute(f'DROP TABLE {table}')
        else:
            con.execute('CREATE TABLE memoria_facts (id INTEGER PRIMARY KEY, value TEXT)')
            con.execute("INSERT INTO memoria_facts VALUES (1, 'Example fact')")

    getattr(store, method)(*args)

    assert len(opened) >= error_connection
    assert_closed(opened)


@pytest.mark.parametrize('method,args,error_connection', READ_CASES)
def test_read_connections_close_on_query_error(tracked_store, monkeypatch, method, args, error_connection):
    store, opened, _ = tracked_store
    real_tables = store._tables

    def fail_query(con):
        if len(opened) == error_connection:
            con.execute('SELECT * FROM missing_connection_test_table')
        return real_tables(con)

    monkeypatch.setattr(store, '_tables', fail_query)
    if method == 'diagnostics':
        result = store.diagnostics()
        assert result['ok'] is False
        assert 'missing_connection_test_table' in result['error']
    else:
        with pytest.raises(sqlite3.OperationalError, match='missing_connection_test_table'):
            getattr(store, method)(*args)

    assert len(opened) == error_connection
    assert_closed(opened)


@pytest.mark.parametrize('method,args,field,expected', WRITE_CASES)
def test_write_connections_commit_then_close(tracked_store, method, args, field, expected):
    store, opened, _ = tracked_store

    result = getattr(store, method)('w1', *args, backup=False)

    assert result['ok'] is True
    # Read through a fresh connection to prove the mutation was committed.
    assert store.get_memory('w1')[field] == expected
    if method == 'supersede_memory':
        replacement = store.get_memory(result['replacement_id'])
        assert replacement['content'] == 'Replacement content'
        assert store.get_memory('w1')['superseded_by'] == replacement['id']
    assert [con.events for con in opened if 'commit' in con.events] == [['commit', 'close']]
    assert_closed(opened)


@pytest.mark.parametrize('method,args,field,expected', WRITE_CASES)
def test_write_connections_rollback_then_close(tracked_store, method, args, field, expected):
    store, opened, real_connect = tracked_store
    with closing(real_connect(store.db_path)) as con, con:
        # Each mutation changes working_memory before reaching episodic_memory.
        # Fail that later update to test rollback of already executed writes,
        # including the replacement INSERT in supersede_memory().
        con.execute("INSERT INTO episodic_memory (id, content) VALUES ('w1', 'Episodic copy')")
        con.execute("""
            CREATE TRIGGER fail_later_update BEFORE UPDATE ON episodic_memory
            BEGIN SELECT RAISE(ABORT, 'forced mutation failure'); END
        """)
        before = {table: con.execute(f'SELECT * FROM {table} ORDER BY id').fetchall()
                  for table in ('working_memory', 'episodic_memory')}

    with pytest.raises(sqlite3.IntegrityError, match='forced mutation failure'):
        getattr(store, method)('w1', *args, backup=False)

    assert [con.events for con in opened if 'rollback' in con.events] == [['rollback', 'close']]
    assert_closed(opened)
    with closing(real_connect(store.db_path)) as con:
        after = {table: con.execute(f'SELECT * FROM {table} ORDER BY id').fetchall()
                 for table in ('working_memory', 'episodic_memory')}
    assert after == before


def test_repeated_wal_reads_close_without_garbage_collection(tracked_store):
    store, opened, real_connect = tracked_store
    with closing(real_connect(store.db_path)) as con:
        assert con.execute('PRAGMA journal_mode=WAL').fetchone()[0] == 'wal'

    for _ in range(220):
        previous_count = len(opened)
        store.stats()
        store.list_memories(limit=10)
        # Retained references prevent GC from rescuing unclosed connections.
        assert_closed(opened[previous_count:])

    assert len(opened) >= 440
    assert_closed(opened)
