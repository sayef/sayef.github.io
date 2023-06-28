# PostgreSQL Installation and DB Creation on Ubuntu 16.04


### Installation

---

- Note: This installation procedure is based on Ubuntu 16.04, for other versions see [https://www.postgresql.org/download/linux/ubuntu/]

1. Create the file /etc/apt/sources.list.d/pgdg.list, and add a line for the repository
   ```sh
   $ echo "deb http://apt.postgresql.org/pub/repos/apt/ xenial-pgdg main" | sudo tee  /etc/apt/sources.list.d/pgdg.list
   ```
2. Import the repository signing key, and update the package lists
   ```sh
   $ wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
   ```
3. Ubuntu includes PostgreSQL by default. To install PostgreSQL on Ubuntu, use the apt-get (or other apt-driving) command
   ```sh
   $ sudo apt-get update
   $ sudo apt-get install postgresql-9.6
   ```
4. The bin directory (`/usr/lib/postgresql/9.6/bin`) of postgresql package contains all the necessary commands. Hence let's make the directory accessible from any where.
   - Edit `~/.profile` file to export postgresql bin directory.
   ```sh
   $ vim ~/.profile
   ```
   - Add the following two lines to export postgresql utility commands
   ```sh
   PATH=$PATH:/usr/lib/postgresql/9.6/bin
   export PATH
   ```
   - Activate the profile
   ```sh
   $ . ~/.profile
   ```
5. During the installation, PostgreSQL will create a default user to operate under. We will be using this user for some administrative purposes.
   - Log in with the following command
   ```sh
   $ sudo su - postgres
   ```
   - Do the same as #4
   - Logout from this user
   ```sh
   $ exit
   ```
6. Now we can execute the utility commands from the shell, even for the `postgres` user.
   ```sh
   $ pg_ctl --help
   $ pg_ctl is a utility to initialize, start, stop, or control a PostgreSQL server.
   ```

### Setup User/Role and DB

---

- Note: PostgreSQL assumes that when we log in, we will be using a username that matches our operating system username, and that we will be connecting to a database with the same name as well.

1. Switch to user postgres
   ```sh
   $ sudo su - postgres
   ```
2. Run `psql` command to create `postgres`'s password. It doesn't have any default password.
   ```sh
   $ psql
   postgres=# \password postgres
   postgres=# ********
   ```
3. Create a user/role for our brand new database, let's say `blog` and give the user to `LOGIN` and `CREATEDB` permissions. Also make the user `OWNER` of the db `blog`
   ```sh
   postgres=# CREATE ROLE admin WITH LOGIN PASSWORD '********';
   postgres=# ALTER ROLE admin CREATEDB;
   postgres=# CREATE DATABASE blog OWNER admin;
   ```
4. Quit `psql` commandline interface
   ```sh
   postgres=# \q
   ```
5. Exit (logout) from this user (`postgres`)
   ```sh
   $ exit
   ```
6. Now we can run `psql` command with that newly created user without switching to user `postgres`
   ```sh
   $ psql --host=localhost --port=5432 --username=admin --password --dbname=blog
   ```
   This will connect the command line interface to the new db `blog` with this user `admin` logged in. Output should be something like the followings:
   ```sh
   psql (9.6.8)
   SSL connection (protocol: TLSv1.2, cipher: ECDHE-RSA-AES256-GCM-SHA384, bits: 256, compression: off)
   Type "help" for help.
   blog=>
   ```

### Creating Cluster Data Directory

---

1. `initdb` will attempt to create the directory we specify if it does not already exist. Of course, this will fail if initdb does not have permissions to write in the parent directory. It's generally recommendable that the PostgreSQL user own not just the data directory but its parent directory as well, so that this should not be a problem. If the desired parent directory doesn't exist either, we will need to create it first, using root privileges if the grandparent directory isn't writable. So the process might look like this:
   ```sh
   $ sudo mkdir /usr/local/pgsql
   $ sudo chown postgres /usr/local/pgsql
   ```
2. Again before running any postgresql commands we need to switch user to `postgres`
   ```sh
   $ sudo su - postgres
   ```
3. Now we can point the directory for creating cluster data directory using `initd`
   ```sh
   $ initdb -D /usr/local/pgsql/data
   ```
   Output should be ended with something like this:
   ```sh
   Success. You can now start the database server using:
   pg_ctl -D /usr/local/pgsql/data -l logfile start
   ```
4. So let's start the server
   ```sh
   $ pg_ctl -D /usr/local/pgsql/data -l pgserver.log start
   ```

