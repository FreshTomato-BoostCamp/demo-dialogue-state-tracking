import sqlite3

import click
from flask import current_app, g
from flask.cli import with_appcontext


def get_db():
    # generate 'g' instance for the first call
    # after that, reuse 'g' for subsequent call
    if 'db' not in g: # g: unique instance for each request
        g.db = sqlite3.connect( # file connection
            current_app.config['DATABASE'], # current app: flask app which raise request
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row # to return as dict

    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()

def init_db():
    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))


@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')

def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)