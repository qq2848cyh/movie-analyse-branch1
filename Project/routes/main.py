"""首页和导航路由"""

from flask import Blueprint, render_template

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def welcome():
    return render_template("welcome.html")


@main_bp.route("/index")
def index():
    return render_template("index.html")


@main_bp.route("/home")
def home():
    return welcome()


@main_bp.route("/aboutMe")
def about_me():
    return render_template("aboutMe.html")
