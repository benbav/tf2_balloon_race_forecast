# A server with at least this many human players is a "lobby": it pings Discord, sets off the website's alarm,
# picks which rotating servers the tracker watches, and is what the forecast and charts count.
# Change it here; the website reads it from site_data.json after the next daily report.
LOBBY_PLAYERS = 10
