/*
Copyright (c) 2016 Ermiya Eskandary & Théophile Cailliau and other contributors
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// ==UserScript==
// @name         Slither.io-bot
// @namespace    http://slither.io/
// @version      1.2.9
// @description  Slither.io bot
// @author       Ermiya Eskandary & Théophile Cailliau
// @match        http://slither.io/
// @grant        none
// ==/UserScript==

/*
Override bot options here
Uncomment variables you wish to change from their default values
Changes you make here will be kept between script versions
*/

var libraries = [];
libraries.push('https://ajax.googleapis.com/ajax/libs/jquery/3.1.0/jquery.min.js');     //JQuery
libraries.push('http://cs.stanford.edu/people/karpathy/convnetjs/build/convnet-min.js');    //convNetJS
libraries.push('http://cs.stanford.edu/people/karpathy/convnetjs/build/deepqlearn.js');     //DQN library
libraries.push('http://cs.stanford.edu/people/karpathy/convnetjs/build/util.js');       // util for DQN

for(var i=0; i<libraries.length; i++){
    var script = document.createElement('script');
    script.src = libraries[i];
    script.type = 'text/javascript';
    document.getElementsByTagName('head')[0].appendChild(script);
}

var customBotOptions = {
    // target fps
    // targetFps: 30,
    // size of arc for collisionAngles
    // arcSize: Math.PI / 8,
    // radius multiple for circle intersects
    // radiusMult: 10,
    // food cluster size to trigger acceleration
    // foodAccelSize: 60,
    // maximum angle of food to trigger acceleration
    // foodAccelAngle:  Math.PI / 3,
    // how many frames per food check
    // foodFrames: 4,
    // round food cluster size up to the nearest
    // foodRoundSize: 5,
    // round food angle up to nearest for angle difference scoring
    // foodRoundAngle: Math.PI / 8,
    // food clusters at or below this size won't be considered
    // if there is a collisionAngle
    // foodSmallSize: 10,
    // angle or higher where enemy heady is considered in the rear
    // rearHeadAngle: 3 * Math.PI / 4,
    // attack emeny rear head at this angle
    // rearHeadDir: Math.PI / 2,
    // quick radius toggle size in approach mode
    // radiusApproachSize: 5,
    // quick radius toggle size in avoid mode
    // radiusAvoidSize: 25,
    // uncomment to quickly revert to the default options
    // if you update the script while this is active,
    // you will lose your custom options
    // useDefaults: true
};

// Custom logging function - disabled by default
window.log = function() {
    if (window.logDebugging) {
        console.log.apply(console, arguments);
    }
};

var canvasUtil = window.canvasUtil = (function() {
    return {
        // Ratio of screen size divided by canvas size.
        canvasRatio: {
            x: window.mc.width / window.ww,
            y: window.mc.height / window.hh
        },

        // Set direction of snake towards the virtual mouse coordinates
        setMouseCoordinates: function(point) {
            //for imitation learning
            bot.currentBotDirection = bot.getSliceIndexFromMouseCoor(point.x,point.y);
            window.xm = point.x;
            window.ym = point.y;
        },

        // Convert snake-relative coordinates to absolute screen coordinates.
        mouseToScreen: function(point) {
            var screenX = point.x + (window.ww / 2);
            var screenY = point.y + (window.hh / 2);
            return {
                x: screenX,
                y: screenY
            };
        },

        // Convert screen coordinates to canvas coordinates.
        screenToCanvas: function(point) {
            var canvasX = window.csc *
                (point.x * canvasUtil.canvasRatio.x) - parseInt(window.mc.style.left);
            var canvasY = window.csc *
                (point.y * canvasUtil.canvasRatio.y) - parseInt(window.mc.style.top);
            return {
                x: canvasX,
                y: canvasY
            };
        },

        // Convert map coordinates to mouse coordinates.
        mapToMouse: function(point) {
            var mouseX = (point.x - window.snake.xx) * window.gsc;
            var mouseY = (point.y - window.snake.yy) * window.gsc;
            return {
                x: mouseX,
                y: mouseY
            };
        },

        // Map coordinates to Canvas coordinates.
        mapToCanvas: function(point) {
            var c = canvasUtil.mapToMouse(point);
            c = canvasUtil.mouseToScreen(c);
            c = canvasUtil.screenToCanvas(c);
            return c;
        },

        // Map to Canvas coordinates conversion for drawing circles.
        circleMapToCanvas: function(circle) {
            var newCircle = canvasUtil.mapToCanvas(circle);
            return canvasUtil.circle(
                newCircle.x,
                newCircle.y,
                // Radius also needs to scale by .gsc
                circle.radius * window.gsc
            );
        },

        // Constructor for point type
        point: function(x, y) {
            var p = {
                x: Math.round(x),
                y: Math.round(y)
            };

            return p;
        },

        // Constructor for rect type
        rect: function(x, y, w, h) {
            var r = {
                x: Math.round(x),
                y: Math.round(y),
                width: Math.round(w),
                height: Math.round(h)
            };

            return r;
        },

        // Constructor for circle type
        circle: function(x, y, r) {
            var c = {
                x: Math.round(x),
                y: Math.round(y),
                radius: Math.round(r)
            };

            return c;
        },

        // Fast atan2
        fastAtan2: function(y, x) {
            const QPI = Math.PI / 4;
            const TQPI = 3 * Math.PI / 4;
            var r = 0.0;
            var angle = 0.0;
            var abs_y = Math.abs(y) + 1e-10;
            if (x < 0) {
                r = (x + abs_y) / (abs_y - x);
                angle = TQPI;
            } else {
                r = (x - abs_y) / (x + abs_y);
                angle = QPI;
            }
            angle += (0.1963 * r * r - 0.9817) * r;
            if (y < 0) {
                return -angle;
            }

            return angle;
        },

        // Adjusts zoom in response to the mouse wheel.
        setZoom: function(e) {
            // Scaling ratio
            if (window.gsc) {
                window.gsc *= Math.pow(0.9, e.wheelDelta / -120 || e.detail / 2 || 0);
            }
        },

        // Restores zoom to the default value.
        resetZoom: function() {
            window.gsc = 0.9;
        },

        // Sets background to the given image URL.
        // Defaults to slither.io's own background.
        setBackground: function(url) {
            url = typeof url !== 'undefined' ? url : '/s/bg45.jpg';
            window.ii.src = url;
        },

        // Draw a rectangle on the canvas.
        drawRect: function(rect, color, fill, alpha) {
            if (alpha === undefined) alpha = 1;

            var context = window.mc.getContext('2d');
            var lc = canvasUtil.mapToCanvas({
                x: rect.x,
                y: rect.y
            });

            context.save();
            context.globalAlpha = alpha;
            context.strokeStyle = color;
            context.rect(lc.x, lc.y, rect.width * window.gsc, rect.height * window.gsc);
            context.stroke();
            if (fill) {
                context.fillStyle = color;
                context.fill();
            }
            context.restore();
        },

        // Draw a circle on the canvas.
        drawCircle: function(circle, color, fill, alpha) {
            if (alpha === undefined) alpha = 1;
            if (circle.radius === undefined) circle.radius = 5;

            var context = window.mc.getContext('2d');
            var drawCircle = canvasUtil.circleMapToCanvas(circle);

            context.save();
            context.globalAlpha = alpha;
            context.beginPath();
            context.strokeStyle = color;
            context.arc(drawCircle.x, drawCircle.y, drawCircle.radius, 0, Math.PI * 2);
            context.stroke();
            if (fill) {
                context.fillStyle = color;
                context.fill();
            }
            context.restore();
        },

        // Draw an angle.
        // @param {number} start -- where to start the angle
        // @param {number} angle -- width of the angle
        // @param {String|CanvasGradient|CanvasPattern} color
        // @param {boolean} fill
        // @param {number} alpha
        drawAngle: function(start, angle, color, fill, alpha) {
            if (alpha === undefined) alpha = 0.6;

            var context = window.mc.getContext('2d');

            context.save();
            context.globalAlpha = alpha;
            context.beginPath();
            context.moveTo(window.mc.width / 2, window.mc.height / 2);
            context.arc(window.mc.width / 2, window.mc.height / 2, window.gsc * 100, start, angle);
            context.lineTo(window.mc.width / 2, window.mc.height / 2);
            context.closePath();
            context.stroke();
            if (fill) {
                context.fillStyle = color;
                context.fill();
            }
            context.restore();
        },

        // Draw a line on the canvas.
        drawLine: function(p1, p2, color, width) {
            if (width === undefined) width = 5;

            var context = window.mc.getContext('2d');
            var dp1 = canvasUtil.mapToCanvas(p1);
            var dp2 = canvasUtil.mapToCanvas(p2);

            context.save();
            context.beginPath();
            context.lineWidth = width * window.gsc;
            context.strokeStyle = color;
            context.moveTo(dp1.x, dp1.y);
            context.lineTo(dp2.x, dp2.y);
            context.stroke();
            context.restore();
        },

        // Given the start and end of a line, is point left.
        isLeft: function(start, end, point) {
            return ((end.x - start.x) * (point.y - start.y) -
                (end.y - start.y) * (point.x - start.x)) > 0;

        },

        // Get distance squared
        getDistance2: function(x1, y1, x2, y2) {
            var distance2 = Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2);
            return distance2;
        },

        getDistance2FromSnake: function(point) {
            point.distance = canvasUtil.getDistance2(window.snake.xx, window.snake.yy,
                point.xx, point.yy);
            return point;
        },

        // Check if point in Rect
        pointInRect: function(point, rect) {
            if (rect.x <= point.x && rect.y <= point.y &&
                rect.x + rect.width >= point.x && rect.y + rect.height >= point.y) {
                return true;
            }
            return false;
        },

        // Check if circles intersect
        circleIntersect: function(circle1, circle2) {
            var bothRadii = circle1.radius + circle2.radius;
            var dx = circle1.x - circle2.x;
            var dy = circle1.y - circle2.y;

            // Pretends the circles are squares for a quick collision check.
            // If it collides, do the more expensive circle check.
            if (dx + bothRadii > 0 && dy + bothRadii > 0 &&
                dx - bothRadii < 0 && dy - bothRadii < 0) {

                var distance2 = canvasUtil.getDistance2(circle1.x, circle1.y, circle2.x, circle2.y);

                if (distance2 < bothRadii * bothRadii) {
                    if (window.visualDebugging) {
                        var collisionPointCircle = canvasUtil.circle(
                            ((circle1.x * circle2.radius) + (circle2.x * circle1.radius)) /
                            bothRadii,
                            ((circle1.y * circle2.radius) + (circle2.y * circle1.radius)) /
                            bothRadii,
                            5
                        );
                        canvasUtil.drawCircle(circle2, 'red', true);
                        canvasUtil.drawCircle(collisionPointCircle, 'cyan', true);
                    }
                    return true;
                }
            }
            return false;
        }
    };
})();

var bot = window.bot = (function() {
    return {
        isBotRunning: false,
        isBotEnabled: true,
        lookForFood: false,
        collisionPoints: [],
        collisionAngles: [],
        scores: [],
        foodTimeout: undefined,
        sectorBoxSide: 0,
        defaultAccel: 0,
        sectorBox: {},
        currentFood: {},
        opt: {
            // These are the bot's default options
            // If you wish to customise these, use
            // customBotOptions above
            targetFps: 30,
            arcSize: Math.PI / 8,
            radiusMult: 10,
            foodAccelSize: 60,
            foodAccelAngle: Math.PI / 3,
            foodFrames: 4,
            foodRoundSize: 5,
            foodRoundAngle: Math.PI / 8,
            foodSmallSize: 10,
            rearHeadAngle: 3 * Math.PI / 4,
            rearHeadDir: Math.PI / 2,
            radiusApproachSize: 5,
            radiusAvoidSize: 25
        },
        MID_X: 0,
        MID_Y: 0,
        MAP_R: 0,

        //This is for ML mode
        NUMBER_OF_SLICES : 32 ,         //MOVEMENT_OFFSET should be Math.PI/(NUMBER_OF_SLICES/2)
        MOVEMENT_OFFSET: Math.PI/16,    //NOTE: the larger the slices the smaller the circle will (because of the comunication speed)
        MOVEMENT_R: 100,
        SEND_C: 0,          //send counter
        ML_mode: 0,
        offsetSize: 35,    //the size of each pixel in label_map is offsetSize^2
        mapSize: 24,        //The label_map size is mapSize^2
        label_map: [],      //represent devision of the game to different sectors
        smallAmountOfFood: 10,
        mediumAmountOfFood: 30,
        emptyLabel: 25,
        selfLabel: 50,
        enemyLabel: 0,
        smallFoodLabel: 100,
        mediumFoodLabel: 150,
        largeFoodLabel: 200,

        //bonus variables
        foodsPenalty: 0,
        enemiesPenalty: 0,
        dCenterPenalty: 0,

        //ML debug vriables
        message_id: 1,
        direction: {x: 0 , y: -100},    //determains the direction of the ML_bot

        //Imatation learning variables
        currentBotDirection: 0,         //recording the action chosen by AL bot translated to direction
        currentBotAcceleration: 0,      //recording if AI bot chose to accelerate or not

        //JS RL learning variabls
        isJSMLInitialized: false,
        convNet: NaN,
        brain: NaN,
        trainingSteps: 500000,
        counterSteps: 0,

        getSnakeWidth: function(sc) {
            if (sc === undefined) sc = window.snake.sc;
            return Math.round(sc * 29.0);
        },

        quickRespawn: function() {
            window.dead_mtm = 0;
            window.login_fr = 0;

            bot.isBotRunning = false;
            window.forcing = true;
            window.connect();
            window.forcing = false;
        },

        // angleBetween - get the smallest angle between two angles (0-pi)
        angleBetween: function(a1, a2) {
            var r1 = 0.0;
            var r2 = 0.0;

            r1 = (a1 - a2) % Math.PI;
            r2 = (a2 - a1) % Math.PI;

            return r1 < r2 ? -r1 : r2;
        },

        // Avoid headPoint
        avoidHeadPoint: function(collisionPoint) {
            var cehang = canvasUtil.fastAtan2(
                collisionPoint.yy - window.snake.yy, collisionPoint.xx - window.snake.xx);
            var diff = bot.angleBetween(window.snake.ehang, cehang);

            if (Math.abs(diff) > bot.opt.rearHeadAngle) {
                var dir = diff > 0 ? -bot.opt.rearHeadDir : bot.opt.rearHeadDir;
                bot.changeHeading(dir);
            } else {
                bot.avoidCollisionPoint(collisionPoint);
            }
        },

        // Change heading by ang
        // +0-pi turn left
        // -0-pi turn right

        changeHeading: function(angle) {
            var heading = {
                x: window.snake.xx + 500 * bot.cos,
                y: window.snake.yy + 500 * bot.sin
            };

            var cos = Math.cos(-angle);
            var sin = Math.sin(-angle);

            window.goalCoordinates = {
                x: Math.round(
                    cos * (heading.x - window.snake.xx) -
                    sin * (heading.y - window.snake.yy) + window.snake.xx),
                y: Math.round(
                    sin * (heading.x - window.snake.xx) +
                    cos * (heading.y - window.snake.yy) + window.snake.yy)
            };

            canvasUtil.setMouseCoordinates(canvasUtil.mapToMouse(window.goalCoordinates));

        },

        // Avoid collision point by ang
        // ang radians <= Math.PI (180deg)
        avoidCollisionPoint: function(collisionPoint, ang) {
            if (ang === undefined || ang > Math.PI) {
                ang = Math.PI;
            }

            var end = {
                x: window.snake.xx + 2000 * bot.cos,
                y: window.snake.yy + 2000 * bot.sin
            };

            if (window.visualDebugging) {
                canvasUtil.drawLine({
                    x: window.snake.xx,
                    y: window.snake.yy
                },
                    end,
                    'orange', 5);
                canvasUtil.drawLine({
                    x: window.snake.xx,
                    y: window.snake.yy
                }, {
                    x: collisionPoint.xx,
                    y: collisionPoint.yy
                },
                    'red', 5);
            }

            var cos = Math.cos(ang);
            var sin = Math.sin(ang);

            if (canvasUtil.isLeft({
                x: window.snake.xx,
                y: window.snake.yy
            }, end, {
                x: collisionPoint.xx,
                y: collisionPoint.yy
            })) {
                sin = -sin;
            }

            window.goalCoordinates = {
                x: Math.round(
                    cos * (collisionPoint.xx - window.snake.xx) -
                    sin * (collisionPoint.yy - window.snake.yy) + window.snake.xx),
                y: Math.round(
                    sin * (collisionPoint.xx - window.snake.xx) +
                    cos * (collisionPoint.yy - window.snake.yy) + window.snake.yy)
            };

            canvasUtil.setMouseCoordinates(canvasUtil.mapToMouse(window.goalCoordinates));
        },

        // Sorting by  property 'distance'
        sortDistance: function(a, b) {
            return a.distance - b.distance;
        },

        // get collision angle index, expects angle +/i 0 to Math.PI
        getAngleIndex: function(angle) {
            const ARCSIZE = bot.opt.arcSize;
            var index;

            if (angle < 0) {
                angle += 2 * Math.PI;
            }

            index = Math.round(angle * (1 / ARCSIZE));

            if (index === (2 * Math.PI) / ARCSIZE) {
                return 0;
            }
            return index;
        },

        // Add to collisionAngles if distance is closer
        addCollisionAngle: function(sp) {
            var ang = canvasUtil.fastAtan2(
                Math.round(sp.yy - window.snake.yy),
                Math.round(sp.xx - window.snake.xx));
            var aIndex = bot.getAngleIndex(ang);

            var actualDistance = Math.round(Math.pow(
                Math.sqrt(sp.distance) - sp.radius, 2));

            if (bot.collisionAngles[aIndex] === undefined) {
                bot.collisionAngles[aIndex] = {
                    x: Math.round(sp.xx),
                    y: Math.round(sp.yy),
                    ang: ang,
                    snake: sp.snake,
                    distance: actualDistance
                };
            } else if (bot.collisionAngles[aIndex].distance > sp.distance) {
                bot.collisionAngles[aIndex].x = Math.round(sp.xx);
                bot.collisionAngles[aIndex].y = Math.round(sp.yy);
                bot.collisionAngles[aIndex].ang = ang;
                bot.collisionAngles[aIndex].snake = sp.snake;
                bot.collisionAngles[aIndex].distance = actualDistance;
            }
        },

        // Get closest collision point per snake.
        getCollisionPoints: function() {
            var scPoint;

            bot.collisionPoints = [];
            bot.collisionAngles = [];

            for (var snake = 0, ls = window.snakes.length; snake < ls; snake++) {
                scPoint = undefined;

                if (window.snakes[snake].id !== window.snake.id &&
                    window.snakes[snake].alive_amt === 1) {

                    scPoint = {
                        xx: window.snakes[snake].xx,
                        yy: window.snakes[snake].yy,
                        snake: snake,
                        radius: bot.getSnakeWidth(window.snakes[snake].sc) / 2
                    };
                    canvasUtil.getDistance2FromSnake(scPoint);
                    bot.addCollisionAngle(scPoint);
                    if (window.visualDebugging) {
                        canvasUtil.drawCircle(canvasUtil.circle(
                                scPoint.xx,
                                scPoint.yy,
                                scPoint.radius),
                            'red', false);
                    }

                    for (var pts = 0, lp = window.snakes[snake].pts.length; pts < lp; pts++) {
                        if (!window.snakes[snake].pts[pts].dying &&
                            canvasUtil.pointInRect({
                                x: window.snakes[snake].pts[pts].xx,
                                y: window.snakes[snake].pts[pts].yy
                            }, bot.sectorBox)
                        ) {
                            var collisionPoint = {
                                xx: window.snakes[snake].pts[pts].xx,
                                yy: window.snakes[snake].pts[pts].yy,
                                snake: snake,
                                radius: bot.getSnakeWidth(window.snakes[snake].sc) / 2
                            };

                            if (window.visualDebugging && true === false) {
                                canvasUtil.drawCircle(canvasUtil.circle(
                                        collisionPoint.xx,
                                        collisionPoint.yy,
                                        collisionPoint.radius),
                                    '#00FF00', false);
                            }

                            canvasUtil.getDistance2FromSnake(collisionPoint);
                            bot.addCollisionAngle(collisionPoint);

                            if (scPoint === undefined ||
                                scPoint.distance > collisionPoint.distance) {
                                scPoint = collisionPoint;
                            }
                        }
                    }
                }
                if (scPoint !== undefined) {
                    bot.collisionPoints.push(scPoint);
                    if (window.visualDebugging) {
                        canvasUtil.drawCircle(canvasUtil.circle(
                            scPoint.xx,
                            scPoint.yy,
                            scPoint.radius
                        ), 'red', false);
                    }
                }
            }

            // WALL
            if (canvasUtil.getDistance2(bot.MID_X, bot.MID_Y, window.snake.xx, window.snake.yy) >
                Math.pow(bot.MAP_R - 1000, 2)) {
                var midAng = canvasUtil.fastAtan2(
                    window.snake.yy - bot.MID_X, window.snake.xx - bot.MID_Y);
                scPoint = {
                    xx: bot.MID_X + bot.MAP_R * Math.cos(midAng),
                    yy: bot.MID_Y + bot.MAP_R * Math.sin(midAng),
                    snake: -1,
                    radius: bot.snakeWidth
                };
                canvasUtil.getDistance2FromSnake(scPoint);
                bot.collisionPoints.push(scPoint);
                bot.addCollisionAngle(scPoint);
                if (window.visualDebugging) {
                    canvasUtil.drawCircle(canvasUtil.circle(
                        scPoint.xx,
                        scPoint.yy,
                        scPoint.radius
                    ), 'yellow', false);
                }
            }

            bot.collisionPoints.sort(bot.sortDistance);
            if (window.visualDebugging) {
                for (var i = 0; i < bot.collisionAngles.length; i++) {
                    if (bot.collisionAngles[i] !== undefined) {
                        canvasUtil.drawLine({
                            x: window.snake.xx,
                            y: window.snake.yy
                        }, {
                            x: bot.collisionAngles[i].x,
                            y: bot.collisionAngles[i].y
                        },
                            '#99ffcc', 2);
                    }
                }
            }
        },

        // Checks to see if you are going to collide with anything in the collision detection radius
        checkCollision: function() {
            var headCircle = canvasUtil.circle(
                window.snake.xx, window.snake.yy,
                bot.speedMult * bot.opt.radiusMult / 2 * bot.snakeRadius
            );

            var fullHeadCircle = canvasUtil.circle(
                window.snake.xx, window.snake.yy,
                bot.opt.radiusMult * bot.snakeRadius
            );

            if (window.visualDebugging) {
                canvasUtil.drawCircle(fullHeadCircle, 'red');
                canvasUtil.drawCircle(headCircle, 'blue', false);
            }

            bot.getCollisionPoints();
            if (bot.collisionPoints.length === 0) return false;

            for (var i = 0; i < bot.collisionPoints.length; i++) {
                var collisionCircle = canvasUtil.circle(
                    bot.collisionPoints[i].xx,
                    bot.collisionPoints[i].yy,
                    bot.collisionPoints[i].radius
                );

                if (canvasUtil.circleIntersect(headCircle, collisionCircle)) {
                    window.setAcceleration(bot.defaultAccel);
                    //for imitation learning
                    bot.currentBotAcceleration = bot.defaultAccel;
                    bot.avoidCollisionPoint(bot.collisionPoints[i]);
                    return true;
                }

                // snake -1 is special case for non snake object.
                if (bot.collisionPoints[i].snake !== -1) {
                    var enemyHeadCircle = canvasUtil.circle(
                        window.snakes[bot.collisionPoints[i].snake].xx,
                        window.snakes[bot.collisionPoints[i].snake].yy,
                        bot.collisionPoints[i].radius
                    );

                    if (canvasUtil.circleIntersect(fullHeadCircle, enemyHeadCircle)) {
                        if (window.snakes[bot.collisionPoints[i].snake].sp > 10) {
                            window.setAcceleration(1);
                            //for imitation learning
                            bot.currentBotAcceleration = 1;

                        } else {
                            window.setAcceleration(bot.defaultAccel);
                            //for imitation learning
                            bot.currentBotAcceleration = bot.defaultAccel;
                        }
                        bot.avoidHeadPoint({
                            xx: window.snakes[bot.collisionPoints[i].snake].xx,
                            yy: window.snakes[bot.collisionPoints[i].snake].yy
                        });
                        return true;
                    }
                }
            }
            window.setAcceleration(bot.defaultAccel);
            //for imitation learning
            bot.currentBotAcceleration = bot.defaultAccel;
            return false;
        },

        sortScore: function(a, b) {
            return b.score - a.score;
        },

        // Round angle difference up to nearest foodRoundAngle degrees.
        // Round food up to nearest foodRoundsz, square for distance^2
        scoreFood: function(f) {
            f.score = Math.pow(Math.ceil(f.sz / bot.opt.foodRoundSize) * bot.opt.foodRoundSize, 2) /
                f.distance / (Math.ceil(f.da / bot.opt.foodRoundAngle) * bot.opt.foodRoundAngle);
        },

        computeFoodGoal: function() {
            var foodClusters = [];
            var foodGetIndex = [];
            var fi = 0;
            var sw = bot.snakeWidth;

            for (var i = 0; i < window.foods.length && window.foods[i] !== null; i++) {
                var a;
                var da;
                var distance;
                var sang = window.snake.ehang;
                var f = window.foods[i];

                if (!f.eaten &&
                    !(
                        canvasUtil.circleIntersect(
                            canvasUtil.circle(f.xx, f.yy, 2),
                            bot.sidecircle_l) ||
                        canvasUtil.circleIntersect(
                            canvasUtil.circle(f.xx, f.yy, 2),
                            bot.sidecircle_r))) {

                    var cx = Math.round(Math.round(f.xx / sw) * sw);
                    var cy = Math.round(Math.round(f.yy / sw) * sw);
                    var csz = Math.round(f.sz);

                    if (foodGetIndex[cx + '|' + cy] === undefined) {
                        foodGetIndex[cx + '|' + cy] = fi;
                        a = canvasUtil.fastAtan2(cy - window.snake.yy, cx - window.snake.xx);
                        da = Math.min(
                            (2 * Math.PI) - Math.abs(a - sang), Math.abs(a - sang));
                        distance = Math.round(
                            canvasUtil.getDistance2(cx, cy, window.snake.xx, window.snake.yy));
                        foodClusters[fi] = {
                            x: cx,
                            y: cy,
                            a: a,
                            da: da,
                            sz: csz,
                            distance: distance,
                            score: 0.0
                        };
                        fi++;
                    } else {
                        foodClusters[foodGetIndex[cx + '|' + cy]].sz += csz;
                    }
                }
            }

            foodClusters.forEach(bot.scoreFood);
            foodClusters.sort(bot.sortScore);

            for (i = 0; i < foodClusters.length; i++) {
                var aIndex = bot.getAngleIndex(foodClusters[i].a);
                if (bot.collisionAngles[aIndex] === undefined ||
                    (Math.sqrt(bot.collisionAngles[aIndex].distance) -
                        bot.snakeRadius * bot.opt.radiusMult / 2 >
                        Math.sqrt(foodClusters[i].distance) &&
                        foodClusters[i].sz > bot.opt.foodSmallSize)
                ) {
                    bot.currentFood = foodClusters[i];
                    return;
                }
            }
            bot.currentFood = {
                x: bot.MID_X,
                y: bot.MID_Y
            };
        },

        foodAccel: function() {
            var aIndex = 0;

            if (bot.currentFood && bot.currentFood.sz > bot.opt.foodAccelSize) {
                aIndex = bot.getAngleIndex(bot.currentFood.a);

                if (
                    bot.collisionAngles[aIndex] && bot.collisionAngles[aIndex].distance >
                    bot.currentFood.distance + bot.snakeWidth * bot.opt.radiusMult &&
                    bot.currentFood.da < bot.opt.foodAccelAngle) {
                    return 1;
                }

                if (bot.collisionAngles[aIndex] === undefined) {
                    return 1;
                }
            }

            return bot.defaultAccel;
        },

        every: function() {
            bot.MID_X = window.grd;
            bot.MID_Y = window.grd;
            bot.MAP_R = window.grd * 0.98;

            bot.sectorBoxSide = Math.floor(Math.sqrt(window.sectors.length)) * window.sector_size;
            bot.sectorBox = canvasUtil.rect(
                window.snake.xx - (bot.sectorBoxSide / 2),
                window.snake.yy - (bot.sectorBoxSide / 2),
                bot.sectorBoxSide, bot.sectorBoxSide);
            // if (window.visualDebugging) canvasUtil.drawRect(bot.sectorBox, '#c0c0c0', true, 0.1);

            bot.cos = Math.cos(window.snake.ang);
            bot.sin = Math.sin(window.snake.ang);

            bot.speedMult = window.snake.sp / 5.78;
            bot.snakeRadius = bot.getSnakeWidth() / 2;
            bot.snakeWidth = bot.getSnakeWidth();

            bot.sidecircle_r = canvasUtil.circle(
                window.snake.lnp.xx -
                ((window.snake.lnp.yy + bot.sin * bot.snakeWidth) -
                    window.snake.lnp.yy),
                window.snake.lnp.yy +
                ((window.snake.lnp.xx + bot.cos * bot.snakeWidth) -
                    window.snake.lnp.xx),
                bot.snakeWidth * bot.speedMult
            );

            bot.sidecircle_l = canvasUtil.circle(
                window.snake.lnp.xx +
                ((window.snake.lnp.yy + bot.sin * bot.snakeWidth) -
                    window.snake.lnp.yy),
                window.snake.lnp.yy -
                ((window.snake.lnp.xx + bot.cos * bot.snakeWidth) -
                    window.snake.lnp.xx),
                bot.snakeWidth * bot.speedMult
            );
        },

        // Main bot
        go: function() {
            bot.every();

            if (bot.checkCollision()) {
                bot.lookForFood = false;
                if (bot.foodTimeout) {
                    window.clearTimeout(bot.foodTimeout);
                    bot.foodTimeout = window.setTimeout(
                        bot.foodTimer, 1000 / bot.opt.targetFps * bot.opt.foodFrames);
                }
            } else {
                bot.lookForFood = true;
                if (bot.foodTimeout === undefined) {
                    bot.foodTimeout = window.setTimeout(
                        bot.foodTimer, 1000 / bot.opt.targetFps * bot.opt.foodFrames);
                }
                window.setAcceleration(bot.foodAccel());
                //for imitation learning
                bot.currentBotAcceleration = bot.foodAccel();
            }
            if(bot.ML_mode == 2){
                if(bot.SEND_C % 10 == 0){
                    bot.sendData()
                }
                bot.SEND_C++;
            }
        },

        // Timer version of food check
        foodTimer: function() {
            if (window.playing && bot.lookForFood &&
                window.snake !== null && window.snake.alive_amt === 1) {
                bot.computeFoodGoal();
                window.goalCoordinates = bot.currentFood;
                canvasUtil.setMouseCoordinates(canvasUtil.mapToMouse(window.goalCoordinates));    //THIS IS THE ORIGINAL CODE!
            }
            bot.foodTimeout = undefined;
        },

        //---------------------------------------------from here starts ML code--------------------------------------------------

        // This function will send a vector of data to the model
        sendData: function(){
            //console.log('Started sendData');
            bot.updateLabelMap();
            var time = new Date();      //for communication debug
            var features = {
                AI_direction: bot.currentBotDirection,
                AI_Acceleration: bot.currentBotAcceleration,
                observation: bot.label_map,
                score: bot.getMyScore(),
                is_dead: window.snake == null,
                //this data is used for debugging the client-server connection
                message_id: bot.message_id,
                hours: time.getHours(),
                minutes: time.getMinutes(),
                seconds: time.getSeconds(),
                bonus: bot.getBonus()
                /*
                //preys: window.preys,
                */
            };
            bot.message_id++;
            $.ajax({
                    type:    "POST",
                    url:     'http://localhost:5000/model',
                    data:    JSON.stringify(features),
                    success: function(data) {
                        //success will only be activated if we are in ML server mode
                        if(bot.ML_mode == 1){
                            if(data.commit_sucide){
                                userInterface.quit();
                                for(var i = 0; i < 25; i++){
                                    //this for is for some time to pass
                                }
                                window.connect();
                            }
                            else{
                                bot.setDirection(data.action);
                                window.setAcceleration(data.do_accelerate);
                            }
                        }
                        //console.log('Got response for request id: '+data.request_id+ ' on '+time.getHours()+':'+time.getMinutes()+':'+time.getSeconds());
                        //console.log('Action chosen: ' + data.action);
                        //console.log('do_accelerate' + data.do_accelerate);

                    },
               // vvv---- This is the new bit
               error:   function(jqXHR, textStatus, errorThrown) {
               alert("Error, status = " + textStatus + ", " +
                     "error thrown: " + errorThrown
                    );
                }
            });

        },

        everyML: function () {
            bot.MAP_R = window.grd * 0.98;
            bot.MID_X = window.grd;
            bot.MID_Y = window.grd;
            //patch - zero acceleration if the snake can't accel;
            if(bot.getMyScore() < 15){
                window.setAcceleration(0);
            }
            //update reward variables
            bot.foodsPenalty = 0;
            bot.enemiesPenalty = 0;
            bot.dCenterPenalty = -Math.sqrt(canvasUtil.getDistance2(window.snake.xx, window.snake.yy, bot.MID_X, bot.MID_Y))*0.00005;
            bot.updateLabelMap();
            if(window.visualDebugging){
                bot.drawNet();
            }
        },

        //this function gets mouse coordinate and converts it to the nearest "slice" index
        getSliceIndexFromMouseCoor:function(x,y){
            var index = 0;
            if(window.snake !== null && window.snake.alive_amt === 1){
                var head = [window.snake.xx, window.snake.yy];
                var theta = Math.atan2(head[1] - y, x - head[0]); // y first!
                theta = (theta + (2*Math.PI)) % (2*Math.PI);//get theta in range[0,2*PI]
                index = (theta + (0.5 * bot.MOVEMENT_OFFSET )) / bot.MOVEMENT_OFFSET;//get the closest slice
                index = Math.floor(index);
                //to account for the opposit directions and bias of atan2 and our mapping
                index = (bot.NUMBER_OF_SLICES + ((bot.NUMBER_OF_SLICES/4) - index)) %bot.NUMBER_OF_SLICES;
            }
            return index;
        },

        getBonus: function () {
            return bot.foodsPenalty + bot.enemiesPenalty + bot.dCenterPenalty + bot.getMyScore()*0.01;
        },

        //This function gets the players current score
        //TODO: when dies, doesn't send the last score. fortunatlly there is a function here that does that (something with get lastScore...)
        getMyScore: function () {
            var divMyScore = document.body.children[17];
            if(divMyScore == undefined ||
                divMyScore.children[0] == undefined ||
                divMyScore.children[0].children[1] == undefined){
                return 0;
            }
            return parseInt(divMyScore.children[0].children[1].innerHTML);
        },

        proccessFoodsPenalty: function (arr) {
            if(arr.length == 0){
                return -1;
            }
            var max = 0;
            var res = 0;
            var r = Math.sqrt(Math.pow(bot.offsetSize*bot.mapSize/2,2)*2);      //pythagoras
            for(var i = arr.length; i--;){
                res += (16.4*r-arr[i]);     //as far as i checked 16.4 is the maximal food size
                if(arr[i] > max){
                    max = arr[i];
                }
            }
            return res/(max*10);
        },

        proccessEnemiesPenalty: function (arr) {
            if(arr.length == 0){
                return 0;
            }
            var min = 0;
            var res = 0;
            var r = Math.sqrt(Math.pow(bot.offsetSize*bot.mapSize/2,2)*2);
            for(var i = arr.length; i--;){
                res += (arr[i] - r);
                if(arr[i] < min){
                    min = arr[i];
                }
            }
            return res/-(min*2.5);
        },


        //creats an nXn label-map, where each pixel in it is at size offsetSize^2
        //n MUST BE EVEN!!!
        restartLabelMap: function(n){
            bot.label_map = new Array(Math.pow(n,2)).fill(bot.emptyLabel);
        },

        //gets x and y coordinates of a point in game unit, and return the closest index
        //to it in the label_map.
        //in case of failure returns -1;
        getIndexFromXY: function(x,y){
            var head = [window.snake.xx, window.snake.yy];
            var index = -1;
            var x_offset, y_offset;
            if((Math.abs(x - head[0]) > bot.mapSize/2 * bot.offsetSize) || (Math.abs(y - head[1]) > bot.mapSize/2 * bot.offsetSize)){
                return -1;
            }
            if(x > head[0]){
                x_offset = bot.mapSize/2 + (Math.floor((x - head[0])/bot.offsetSize));
            }
            else {
                x_offset = bot.mapSize/2 + (Math.ceil((x - head[0])/bot.offsetSize));
            }
            if(y < head[1]){
                y_offset = bot.mapSize/2 + (Math.floor((head[1] - y)/bot.offsetSize));
            }
            else{
                y_offset = bot.mapSize/2 + (Math.ceil((head[1] - y)/bot.offsetSize));
            }

            index = bot.mapSize*y_offset + x_offset;
            return index;
        },

        updateLabelMap: function () {
            bot.restartLabelMap(bot.mapSize);
            bot.labelMapByFoods();
            bot.labelMapBySelf();
            bot.lableMapBySnakes();
            bot.labelByEdge();
        },

        labelByEdge: function(){
            if (window.snake != null && canvasUtil.getDistance2(bot.MID_X, bot.MID_Y, window.snake.xx, window.snake.yy) >
                Math.pow(bot.MAP_R - 750, 2)){
                bot.markEdge();
                }
        },

        markEdge: function () {
            //console.log('entered to markEdge');
            var r_location = [];
            var teta = 0;
            var cp = {};     //current point
            var to = 0;     //teta offset
            var index = 0;

            //get my location relatively to map center
            r_location = [window.snake.xx - bot.MID_X, bot.MID_Y - window.snake.yy];
            //get teta using atan2, notice it gets the y coordinate first
            teta = Math.atan2(r_location[1],r_location[0]);
            //get closest point to me on the edge
            cp = {
                x: Math.cos(teta)*bot.MAP_R  + bot.MID_X,
                y: bot.MID_Y - Math.sin(teta)*bot.MAP_R
            };
            index = bot.getIndexFromXY(cp.x, cp.y);

            bot.label_map[index] = bot.enemyLabel;
            //calculate offset to move along the edge
            to = bot.offsetSize/bot.MAP_R;
            //mark clockwise and the counter clockwise.
            bot.markClockwise(teta,to);
            bot.markClockwise(teta,-to);
            //console.log('finished to markEdge');
        },

        //input: the teta to current point and teta offset
        markClockwise: function (teta, to) {
            //console.log('started markClockwise');
            var index = 0;
            var k = 1;
            var np = {};        //next point
            //go clockwise from current and label points on label_map, until you are out of label_map
            while(index >= 0){
                //get new np using offset
                if(k != 1){
                    bot.label_map[index] = bot.enemyLabel;
                }
                np = {
                    x: Math.cos(teta + k*to)*bot.MAP_R + bot.MID_X,
                    y: bot.MID_Y - Math.sin(teta + k*to)*bot.MAP_R
                };
                index = bot.getIndexFromXY(np.x,np.y);

                k++;
            }
            //console.log('finished markClockwise');
        },

        //Updates all the points in label_map which are close to enemy snakes
        lableMapBySnakes: function(){
            var indexes = [];
            var cp = {};        //center point
            var self = [0,0];
            var penalties = [];
            if(window.snake !== null && window.snake.alive_amt === 1){
                self = [window.snake.xx, window.snake.yy];
            }
            for (var snake = 0, ls = window.snakes.length; snake < ls; snake++) {
                if (window.snakes[snake].id !== window.snake.id &&
                    window.snakes[snake].alive_amt === 1) {
                    for (var pt = 0, pts = window.snakes[snake].pts.length; pt < pts; pt++){
                        if(!window.snakes[snake].pts[pt].dying){
                            cp = {
                                x: window.snakes[snake].pts[pt].xx,
                                y: window.snakes[snake].pts[pt].yy,
                                r: bot.getSnakeWidth(window.snakes[snake].sc) / 2
                            };

                            //appending all possible indexes
                            indexes.push(bot.getIndexFromXY(cp.x,cp.y));
                            indexes.push(bot.getIndexFromXY(cp.x + cp.r,cp.y));
                            indexes.push(bot.getIndexFromXY(cp.x - cp.r,cp.y));
                            indexes.push(bot.getIndexFromXY(cp.x,cp.y + cp.r));
                            indexes.push(bot.getIndexFromXY(cp.x,cp.y - cp.r));

                        }
                        for(var i = 0; i < indexes.length; i++){
                            if (!((indexes[i] < 0) || indexes[i] > Math.pow(bot.mapSize,2))){
                                bot.label_map[indexes[i]] = bot.enemyLabel;
                                if(i==0){
                                    penalties.push(-Math.sqrt(canvasUtil.getDistance2(self[0],self[1],cp.x,cp.y)));
                                }
                            }
                        }
                        indexes = [];
                    }
                }
            }
            bot.enemiesPenalty = bot.proccessEnemiesPenalty(penalties);
        },

        //Labels the label map according to nearby foods
        labelMapByFoods: function () {
            var index = -1;
            var self = [0,0];
            var penalties = [];
            if(window.snake !== null && window.snake.alive_amt === 1){
                self = [window.snake.xx, window.snake.yy];
            }
            //calculates the size of the foods near a point on label_map
            var foodSums = new Array(Math.pow(bot.mapSize,2)).fill(0);
            for (var i = 0; i < window.foods.length && window.foods[i] !== null; i++) {
                index = bot.getIndexFromXY(window.foods[i].xx,window.foods[i].yy);
                if(index < 0 || index > Math.pow(bot.mapSize,2)){
                    continue;
                }
                foodSums[index] += window.foods[i].sz;
                penalties.push(Math.sqrt(canvasUtil.getDistance2(self[0],self[1],window.foods[i].xx,window.foods[i].yy))*window.foods[i].sz);
            }
            //label each point in label_map
            for(i = 0; i<bot.label_map.length; i++){
                if(foodSums[i] == 0){
                    bot.label_map[i] = bot.emptyLabel;
                }
                else if(foodSums[i] < bot.smallAmountOfFood){
                    bot.label_map[i] = bot.smallFoodLabel;

                }
                else if(foodSums[i] < bot.mediumAmountOfFood){
                    bot.label_map[i] = bot.mediumFoodLabel;
                }
                else{
                    foodSums[i] = bot.largeFoodLabel;
                }
            }
            bot.foodsPenalty = bot.proccessFoodsPenalty(penalties);
        },

        //label label_map by selfs body.
        labelMapBySelf: function() {
            if(window.snake !== null && window.snake.alive_amt === 1){
                var indexes = [];
                var cp = {};        //center point
                for(var i = 0; i < window.snake.pts.length; i++){
                    if(!window.snake.pts[i].dying){
                        cp = {
                                x: window.snake.pts[i].xx,
                                y: window.snake.pts[i].yy,
                                r: bot.getSnakeWidth(window.snake.sc) / 2
                            };

                        //appending all possible indexes
                        indexes.push(bot.getIndexFromXY(cp.x,cp.y));
                        indexes.push(bot.getIndexFromXY(cp.x + cp.r,cp.y));
                        indexes.push(bot.getIndexFromXY(cp.x - cp.r,cp.y));
                        indexes.push(bot.getIndexFromXY(cp.x,cp.y + cp.r));
                        indexes.push(bot.getIndexFromXY(cp.x,cp.y - cp.r));

                        for(var j = 0; j < indexes.length; j++){
                            if(indexes[j] < 0 || indexes[j] > Math.pow(bot.mapSize,2)){
                                continue;
                            }
                            bot.label_map[indexes[j]] = bot.selfLabel;
                        }
                    }
                }
            }
        },

        //If side == 1 move right, if side == 2 move left, else do nothing
        //the angle is in radians!
        move: function(side){
            var angle = Math.atan2(bot.direction.y, bot.direction.x);
            if(side == 1){
                angle += bot.MOVEMENT_OFFSET/10;
            }
            else if(side ==2){
                angle -= bot.MOVEMENT_OFFSET/10;
            }
            bot.direction = {x: bot.MOVEMENT_R*Math.cos(angle),
                                 y: bot.MOVEMENT_R*Math.sin(angle)};
            canvasUtil.setMouseCoordinates(bot.direction);

        },

        //moves the bot in the direction of the i'th slice of a circle.
        //this circle is indexed CLOCKWISE were 0 is 00:00 o'clock
        setDirection: function (sliceIndex) {
            bot.direction = {x: -bot.MOVEMENT_R*Math.cos(sliceIndex*bot.MOVEMENT_OFFSET + Math.PI/2),
                                 y: bot.MOVEMENT_R*Math.sin(sliceIndex*bot.MOVEMENT_OFFSET - Math.PI/2)};

            //adding vibration that make sure the bot will move in the right direction
            var r = Math.random();
            if(r < 0.33){
                bot.move(1);
            }
            else if(r < 0.66){
                bot.move(2);
            }
            canvasUtil.setMouseCoordinates(bot.direction);
        },

        //VISUAL DEBUGGER code starts here
        drawNet: function(){
            //var localMap = bot.label_map;
            var head = [window.snake.xx, window.snake.yy];
            for(var i = 0; i < bot.label_map.length; i++){
                var p = bot.getPointFromIndex(i,head);
                if(bot.label_map[i] == bot.enemyLabel ){
                    canvasUtil.drawCircle(canvasUtil.circle(p.x, p.y,7),
                                    'red', true);
                    continue;
                }
                else if(bot.label_map[i] == bot.selfLabel){
                    canvasUtil.drawCircle(canvasUtil.circle(p.x, p.y,4),
                                    'purple', true);
                    continue;
                }
                else if(bot.label_map[i] != bot.emptyLabel ){
                    canvasUtil.drawRect(canvasUtil.rect(p.x, p.y,11,11),
                                        'lawngreen', true);
                    continue;
                }
                else{
                    canvasUtil.drawCircle(canvasUtil.circle(p.x, p.y,0.5),'midnightblue', true);
                }

            }
        },

        getPointFromIndex: function (index, head) {
            var n = Math.sqrt(bot.label_map.length,2);
            return {
                x: head[0] + (index%n - n/2)*bot.offsetSize,
                y: head[1] + (n/2 - Math.floor(index/n))*bot.offsetSize
            }
        },
        
        createConvNet: function () {
            var num_actions = bot.NUMBER_OF_SLICES*2;
            var layer_defs = [];
            layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth: bot.label_map.length});
            layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
            layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
            layer_defs.push({type:'regression', num_neurons:num_actions});

            //This was an attempt to create a CNN to the JS-ML code, but from some reasons it caused hugh lags
            /*
            //input: label map, four frames
            layer_defs.push({type:'input', out_sx: bot.mapSize, out_sy: bot.mapSize, out_depth:1});
            //conv_layer_1 + polling
            layer_defs.push({type:'conv', sx:6, filters:16, stride:1, pad:2, activation:'relu'});
            layer_defs.push({type:'pool', sx:2, stride:2});
            //conv_layer_2 + polling
            layer_defs.push({type:'conv', sx:6, filters:32, stride:1, pad:2, activation:'relu'});
            layer_defs.push({type:'pool', sx:2, stride:2});
            //fully connected layers:
            layer_defs.push({type:'fc', num_neurons: 256, activation:'relu'});
            layer_defs.push({type:'regression', num_neurons:num_actions});
            //var net = new convnetjs.Net;
            //net.makeLayers(layer_defs);
            */
            return layer_defs;
        },

        //Returns the RL brain. most of this code is taken from this tutorial: http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html
        createRLBrain: function () {
            var num_inputs = Math.pow(bot.mapSize,2);
            var num_actions = bot.NUMBER_OF_SLICES*2; // number of possible directions with or withour acceleration
            var temporal_window = 0; // amount of temporal memory. 0 = agent lives in-the-moment :)
            var network_size = num_inputs*temporal_window + num_actions*temporal_window + num_inputs;

            // options for the Temporal Difference learner that trains the above net
            // by backpropping the temporal difference learning rule.
            var tdtrainer_options = {learning_rate:0.001, momentum:0.0, batch_size:64, l2_decay:0.01};

            var opt = {};
            opt.temporal_window = temporal_window;
            opt.experience_size = 100000;
            opt.start_learn_threshold = 10000;
            opt.gamma = 0.9;
            opt.learning_steps_total = bot.trainingSteps;
            opt.learning_steps_burnin = 3000;
            opt.epsilon_min = 0.05;
            opt.epsilon_test_time = 0.05;
            opt.layer_defs = bot.convNet;
            opt.tdtrainer_options = tdtrainer_options;

            var brain = new deepqlearn.Brain(num_inputs, num_actions, opt);
            return brain;
        },

        //source: https://stackoverflow.com/questions/4492385/how-to-convert-simple-array-into-two-dimensional-arraymatrix-in-javascript-or
        listToMatrix: function(list, elementsPerSubArray){
            var matrix = [], i, k;

            for (i = 0, k = -1; i < list.length; i++) {
                if (i % elementsPerSubArray === 0) {
                    k++;
                    matrix[k] = [];
                }

                matrix[k].push(list[i]);
            }

            return matrix;
        }


    };
})();

var userInterface = window.userInterface = (function() {
    // Save the original slither.io functions so we can modify them, or reenable them later.
    var original_keydown = document.onkeydown;
    var original_onmouseDown = window.onmousedown;
    var original_oef = window.oef;
    var original_redraw = window.redraw;
    var original_onmousemove = window.onmousemove;

    window.oef = function() {};
    window.redraw = function() {};

    // Modify the redraw()-function to remove the zoom altering code
    // and replace b.globalCompositeOperation = "lighter"; to "hard-light".
    var original_redraw_string = original_redraw.toString();
    var new_redraw_string = original_redraw_string.replace(
        'gsc!=f&&(gsc<f?(gsc+=2E-4,gsc>=f&&(gsc=f)):(gsc-=2E-4,gsc<=f&&(gsc=f)))', '');
    new_redraw_string = new_redraw_string.replace(/b.globalCompositeOperation="lighter"/gi,
        'b.globalCompositeOperation="hard-light"');
    var new_redraw = new Function(new_redraw_string.substring(
        new_redraw_string.indexOf('{') + 1, new_redraw_string.lastIndexOf('}')));

    return {
        overlays: {},

        initOverlays: function() {
            var botOverlay = document.createElement('div');
            botOverlay.style.position = 'fixed';
            botOverlay.style.right = '5px';
            botOverlay.style.bottom = '112px';
            botOverlay.style.width = '150px';
            botOverlay.style.height = '85px';
            // botOverlay.style.background = 'rgba(0, 0, 0, 0.5)';
            botOverlay.style.color = '#C0C0C0';
            botOverlay.style.fontFamily = 'Consolas, Verdana';
            botOverlay.style.zIndex = 999;
            botOverlay.style.fontSize = '14px';
            botOverlay.style.padding = '5px';
            botOverlay.style.borderRadius = '5px';
            botOverlay.className = 'nsi';
            document.body.appendChild(botOverlay);

            var serverOverlay = document.createElement('div');
            serverOverlay.style.position = 'fixed';
            serverOverlay.style.right = '5px';
            serverOverlay.style.bottom = '5px';
            serverOverlay.style.width = '160px';
            serverOverlay.style.height = '14px';
            serverOverlay.style.color = '#C0C0C0';
            serverOverlay.style.fontFamily = 'Consolas, Verdana';
            serverOverlay.style.zIndex = 999;
            serverOverlay.style.fontSize = '14px';
            serverOverlay.className = 'nsi';
            document.body.appendChild(serverOverlay);

            var prefOverlay = document.createElement('div');
            prefOverlay.style.position = 'fixed';
            prefOverlay.style.left = '10px';
            prefOverlay.style.top = '75px';
            prefOverlay.style.width = '260px';
            prefOverlay.style.height = '210px';
            // prefOverlay.style.background = 'rgba(0, 0, 0, 0.5)';
            prefOverlay.style.color = '#C0C0C0';
            prefOverlay.style.fontFamily = 'Consolas, Verdana';
            prefOverlay.style.zIndex = 999;
            prefOverlay.style.fontSize = '14px';
            prefOverlay.style.padding = '5px';
            prefOverlay.style.borderRadius = '5px';
            prefOverlay.className = 'nsi';
            document.body.appendChild(prefOverlay);

            var statsOverlay = document.createElement('div');
            statsOverlay.style.position = 'fixed';
            statsOverlay.style.left = '10px';
            statsOverlay.style.top = '340px';
            statsOverlay.style.width = '140px';
            statsOverlay.style.height = '210px';
            // statsOverlay.style.background = 'rgba(0, 0, 0, 0.5)';
            statsOverlay.style.color = '#C0C0C0';
            statsOverlay.style.fontFamily = 'Consolas, Verdana';
            statsOverlay.style.zIndex = 998;
            statsOverlay.style.fontSize = '14px';
            statsOverlay.style.padding = '5px';
            statsOverlay.style.borderRadius = '5px';
            statsOverlay.className = 'nsi';
            document.body.appendChild(statsOverlay);

            userInterface.overlays.botOverlay = botOverlay;
            userInterface.overlays.serverOverlay = serverOverlay;
            userInterface.overlays.prefOverlay = prefOverlay;
            userInterface.overlays.statsOverlay = statsOverlay;
        },

        toggleOverlays: function() {
            Object.keys(userInterface.overlays).forEach(function(okey) {
                var oVis = userInterface.overlays[okey].style.visibility !== 'hidden' ?
                    'hidden' : 'visible';
                userInterface.overlays[okey].style.visibility = oVis;
                window.visualDebugging = oVis === 'visible';
            });
        },
        toggleLeaderboard: function() {
            window.leaderboard = !window.leaderboard;
            window.log('Leaderboard set to: ' + window.leaderboard);
            userInterface.savePreference('leaderboard', window.leaderboard);
            if (window.leaderboard) {
                // window.lbh.style.display = 'block';
                // window.lbs.style.display = 'block';
                // window.lbp.style.display = 'block';
                window.lbn.style.display = 'block';
            } else {
                // window.lbh.style.display = 'none';
                // window.lbs.style.display = 'none';
                // window.lbp.style.display = 'none';
                window.lbn.style.display = 'none';
            }
        },
        removeLogo: function() {
            if (typeof window.showlogo_iv !== 'undefined') {
                window.ncka = window.lgss = window.lga = 1;
                clearInterval(window.showlogo_iv);
                showLogo(true);
            }
        },
        // Save variable to local storage
        savePreference: function(item, value) {
            window.localStorage.setItem(item, value);
            userInterface.onPrefChange();
        },

        // Load a variable from local storage
        loadPreference: function(preference, defaultVar) {
            var savedItem = window.localStorage.getItem(preference);
            if (savedItem !== null) {
                if (savedItem === 'true') {
                    window[preference] = true;
                } else if (savedItem === 'false') {
                    window[preference] = false;
                } else {
                    window[preference] = savedItem;
                }
                window.log('Setting found for ' + preference + ': ' + window[preference]);
            } else {
                window[preference] = defaultVar;
                window.log('No setting found for ' + preference +
                    '. Used default: ' + window[preference]);
            }
            userInterface.onPrefChange();
            return window[preference];
        },

        // Saves username when you click on "Play" button
        playButtonClickListener: function() {
            userInterface.saveNick();
            userInterface.loadPreference('autoRespawn', false);
            userInterface.onPrefChange();
        },

        // Preserve nickname
        saveNick: function() {
            var nick = document.getElementById('nick').value;
            userInterface.savePreference('savedNick', nick);
        },

        // Hide top score
        hideTop: function() {
            var nsidivs = document.querySelectorAll('div.nsi');
            for (var i = 0; i < nsidivs.length; i++) {
                if (nsidivs[i].style.top === '4px' && nsidivs[i].style.width === '300px') {
                    nsidivs[i].style.visibility = 'hidden';
                    bot.isTopHidden = true;
                    window.topscore = nsidivs[i];
                }
            }
        },

        // Store FPS data
        framesPerSecond: {
            fps: 0,
            fpsTimer: function() {
                if (window.playing && window.fps && window.lrd_mtm) {
                    if (Date.now() - window.lrd_mtm > 970) {
                        userInterface.framesPerSecond.fps = window.fps;
                    }
                }
            }
        },

        onkeydown: function(e) {
            // Original slither.io onkeydown function + whatever is under it
            original_keydown(e);
            if (window.playing) {
                // Letter `T` to toggle bot
                if (e.keyCode === 84) {
                    bot.isBotEnabled = !bot.isBotEnabled;
                }
                // Letter 'U' to toggle debugging (console)
                if (e.keyCode === 85) {
                    window.logDebugging = !window.logDebugging;
                    window.log('Log debugging set to: ' + window.logDebugging);
                    userInterface.savePreference('logDebugging', window.logDebugging);
                }
                // Letter 'Y' to toggle debugging (visual)
                if (e.keyCode === 89) {
                    window.visualDebugging = !window.visualDebugging;
                    window.log('Visual debugging set to: ' + window.visualDebugging);
                    userInterface.savePreference('visualDebugging', window.visualDebugging);
                }
                // Letter 'G' to toggle leaderboard
                if (e.keyCode === 71) {
                    userInterface.toggleLeaderboard(!window.leaderboard);
                }
                // Letter 'I' to toggle autorespawn
                if (e.keyCode === 73) {
                    window.autoRespawn = !window.autoRespawn;
                    window.log('Automatic Respawning set to: ' + window.autoRespawn);
                    userInterface.savePreference('autoRespawn', window.autoRespawn);
                }

                // Letter 'M' to toggle Machine learning modes
                if (e.keyCode === 77) {
                    bot.ML_mode = (bot.ML_mode + 1)%4;
                    window.log('Machine learning mode set to: ' + bot.ML_mode);
                    userInterface.savePreference('MachineLearning', bot.ML_mode);
                }

                // Letter 'H' to toggle hidden mode
                if (e.keyCode === 72) {
                    userInterface.toggleOverlays();
                }
                // Letter 'B' to prompt for a custom background url
                if (e.keyCode === 66) {
                    var url = prompt('Please enter a background url:');
                    if (url !== null) {
                        canvasUtil.setBackground(url);
                    }
                }
                // Letter 'O' to change rendermode (visual)
                if (e.keyCode === 79) {
                    userInterface.toggleMobileRendering(!window.mobileRender);
                }
                // Letter 'A' to increase collision detection radius
                if (e.keyCode === 65) {
                    bot.opt.radiusMult++;
                    window.log(
                        'radiusMult set to: ' + bot.opt.radiusMult);
                }
                // Letter 'S' to decrease collision detection radius
                if (e.keyCode === 83) {
                    if (bot.opt.radiusMult > 1) {
                        bot.opt.radiusMult--;
                        window.log(
                            'radiusMult set to: ' +
                            bot.opt.radiusMult);
                    }
                }
                // Letter 'D' to quick toggle collision radius
                if (e.keyCode === 68) {
                    if (bot.opt.radiusMult >
                        ((bot.opt.radiusAvoidSize - bot.opt.radiusApproachSize) /
                            2 + bot.opt.radiusApproachSize)) {
                        bot.opt.radiusMult = bot.opt.radiusApproachSize;
                    } else {
                        bot.opt.radiusMult = bot.opt.radiusAvoidSize;
                    }
                    window.log(
                        'radiusMult set to: ' + bot.opt.radiusMult);
                }
                // Letter 'Z' to reset zoom
                if (e.keyCode === 90) {
                    canvasUtil.resetZoom();
                }
                // Letter 'Q' to quit to main menu
                if (e.keyCode === 81) {
                    window.autoRespawn = false;
                    userInterface.quit();
                }
                // 'ESC' to quickly respawn
                if (e.keyCode === 27) {
                    bot.quickRespawn();
                }
                // Save nickname when you press "Enter"
                if (e.keyCode === 13) {
                    userInterface.saveNick();
                }
                userInterface.onPrefChange();
            }
        },

        onmousedown: function(e) {
            if (window.playing) {
                switch (e.which) {
                    // "Left click" to manually speed up the slither
                    case 1:
                        bot.defaultAccel = 1;
                        if (!bot.isBotEnabled) {
                            original_onmouseDown(e);
                        }
                        break;
                        // "Right click" to toggle bot in addition to the letter "T"
                    case 3:
                        bot.isBotEnabled = !bot.isBotEnabled;
                        break;
                }
            } else {
                original_onmouseDown(e);
            }
            userInterface.onPrefChange();
        },

        onmouseup: function() {
            bot.defaultAccel = 0;
        },

        // Manual mobile rendering
        toggleMobileRendering: function(mobileRendering) {
            window.mobileRender = mobileRendering;
            window.log('Mobile rendering set to: ' + window.mobileRender);
            userInterface.savePreference('mobileRender', window.mobileRender);
            // Set render mode
            if (window.mobileRender) {
                window.render_mode = 1;
                window.want_quality = 0;
                window.high_quality = false;
            } else {
                window.render_mode = 2;
                window.want_quality = 1;
                window.high_quality = true;
            }
        },

        // Update stats overlay.
        updateStats: function() {
            var oContent = [];
            var median;

            if (    bot.scores.length === 0) return;
            median = Math.round((bot.scores[Math.floor((bot.scores.length - 1) / 2)] +
                     bot.scores[Math.ceil((bot.scores.length - 1) / 2)]) / 2);

            oContent.push('games played: ' + bot.scores.length);
            oContent.push('a: ' + Math.round(
                bot.scores.reduce(function(a, b) { return a + b; }) / (bot.scores.length)) +
                ' m: ' + median);

            for (var i = 0; i < bot.scores.length && i < 10; i++) {
                oContent.push(i + 1 + '. ' + bot.scores[i]);
            }

            userInterface.overlays.statsOverlay.innerHTML = oContent.join('<br/>');
        },

        onPrefChange: function() {
            // Set static display options here.
            var oContent = [];
            var ht = userInterface.handleTextColor;

            oContent.push('version: ' + GM_info.script.version);
            oContent.push('[T / Right click] bot: ' + ht(bot.isBotEnabled));
            oContent.push('[O] mobile rendering: ' + ht(window.mobileRender));
            oContent.push('[A/S] radius multiplier: ' + bot.opt.radiusMult);
            oContent.push('[D] quick radius change ' +
                bot.opt.radiusApproachSize + '/' + bot.opt.radiusAvoidSize);
            oContent.push('[I] auto respawn: ' + ht(window.autoRespawn));
            oContent.push('[M] ML mode: ' + userInterface.handleMLTextColor(bot.ML_mode));
            oContent.push('[G] leaderboard overlay: ' + ht(window.leaderboard));
            oContent.push('[Y] visual debugging: ' + ht(window.visualDebugging));
            oContent.push('[U] log debugging: ' + ht(window.logDebugging));
            oContent.push('[H] overlays');
            oContent.push('[B] change background');
            oContent.push('[Mouse Wheel] zoom');
            oContent.push('[Z] reset zoom');
            oContent.push('[ESC] quick respawn');
            oContent.push('[Q] quit to menu');

            userInterface.overlays.prefOverlay.innerHTML = oContent.join('<br/>');
        },

        onFrameUpdate: function() {
            // Botstatus overlay
            var oContent = [];

            if (window.playing && window.snake !== null) {
                oContent.push('fps: ' + userInterface.framesPerSecond.fps);

                // Display the X and Y of the snake
                oContent.push('x: ' +
                    (Math.round(window.snake.xx) || 0) + ' y: ' +
                    (Math.round(window.snake.yy) || 0));

                if (window.goalCoordinates) {
                    oContent.push('target');
                    oContent.push('x: ' + window.goalCoordinates.x + ' y: ' +
                        window.goalCoordinates.y);
                    if (window.goalCoordinates.sz) {
                        oContent.push('sz: ' + window.goalCoordinates.sz);
                    }
                }

                if (window.bso !== undefined && userInterface.overlays.serverOverlay.innerHTML !==
                    window.bso.ip + ':' + window.bso.po) {
                    userInterface.overlays.serverOverlay.innerHTML =
                        window.bso.ip + ':' + window.bso.po;
                }
            }

            userInterface.overlays.botOverlay.innerHTML = oContent.join('<br/>');

            if (window.playing && window.visualDebugging && !(bot.ML_mode == 1 || bot.ML_mode == 3)) {
                // Only draw the goal when a bot has a goal.
                if (window.goalCoordinates && bot.isBotEnabled) {
                    var headCoord = {
                        x: window.snake.xx,
                        y: window.snake.yy
                    };
                    canvasUtil.drawLine(
                        headCoord,
                        window.goalCoordinates,
                        'green');
                    canvasUtil.drawCircle(window.goalCoordinates, 'red', true);
                }
            }
        },

        oefTimer: function() {
            var start = Date.now();
            // Original slither.io oef function + whatever is under it
            original_oef();
            // Modified slither.io redraw function
            new_redraw();

            if (window.playing && bot.isBotEnabled && window.snake !== null) {
                window.onmousemove = function() {};
                bot.isBotRunning = true;
                //switch between ML mode an AI mode
                if(bot.ML_mode == 1){
                    bot.everyML();
                    //This is how we control the amount of requests per time
                    if(bot.SEND_C % 10 == 0){
                        bot.sendData();
                    }
                    bot.SEND_C++;
                }
                else if(bot.ML_mode == 3){
                    bot.everyML();
                    if(!bot.isJSMLInitialized){
                        //initialise bot JS ML variables:
                        console.log("Initialized JS ML variables!");
                        bot.convNet = bot.createConvNet();
                        bot.brain = bot.createRLBrain();
                        bot.isJSMLInitialized = true;
                    }
                    var action = bot.brain.forward(bot.label_map);
                    bot.setDirection(action%bot.NUMBER_OF_SLICES);
                    window.setAcceleration(Math.floor(action/bot.NUMBER_OF_SLICES));
                    if(bot.counterSteps < bot.trainingSteps){
                        if(bot.counterSteps%1000 == 0){
                            console.log("keep training! " + bot.counterSteps + '/' + bot.trainingSteps + ' steps done!');
                        }
                        bot.brain.backward([bot.getMyScore()]);
                        bot.counterSteps++;
                    }
                    else if(bot.counterSteps == bot.trainingSteps){
                        console.log("finished training!");
                        bot.brain.epsilon_test_time = 0.0;
                        bot.brain.learning = false;
                        bot.counterSteps++;
                    }
                }
                else{
                    bot.go();
                }
            }
            //snake died.
            else if (bot.isBotEnabled && bot.isBotRunning) {
                //console.log('snake died');      //for debug
                bot.isBotRunning = false;
                if(bot.ML_mode == 1 || bot.ML_mode == 2){
                    //send message 3 times:
                    for(var i=0; i<3; i++){
                        bot.sendData();
                    }
                }
                if (window.lastscore && window.lastscore.childNodes[1]) {
                    bot.scores.push(parseInt(window.lastscore.childNodes[1].innerHTML));
                    bot.scores.sort(function(a, b) {
                        return b - a;
                    });
                    userInterface.updateStats();
                }

                if (window.autoRespawn) {
                    window.connect();
                }
            }

            if (!bot.isBotEnabled || !bot.isBotRunning) {
                window.onmousemove = original_onmousemove;
            }

            userInterface.onFrameUpdate();
            setTimeout(userInterface.oefTimer, (1000 / bot.opt.targetFps) - (Date.now() - start));
        },

        // Quit to menu
        quit: function() {
            if (window.playing && window.resetGame) {
                window.want_close_socket = true;
                window.dead_mtm = 0;
                if (window.play_btn) {
                    window.play_btn.setEnabled(true);
                }
                window.resetGame();
            }
        },

        // Update the relation between the screen and the canvas.
        onresize: function() {
            window.resize();
            // Canvas different size from the screen (often bigger).
            canvasUtil.canvasRatio = {
                x: window.mc.width / window.ww,
                y: window.mc.height / window.hh
            };
        },
        // Handles the text color of the bot preferences
        // enabled = green
        // disabled = red
        handleTextColor: function(enabled) {
            return '<span style=\"color:' +
                (enabled ? 'green;\">enabled' : 'red;\">disabled') + '</span>';
        },

        handleMLTextColor: function() {
            var res = '<span style=\"color:';
            switch(bot.ML_mode){
                case 1:
                    res = res + 'blue;\">ML server mode' + '</span>';
                    break;
                case 2:
                    res = res + 'yellow;\">IL mode' + '</span>';
                    break;
                case 3:
                    res = res + 'LawnGreen;\">JS ML mode' + '</span>';
                    break;
                default:
                    res = res + 'red;\">disabled' + '</span>';
            }
            return res;
        }
    };
})();

// Main
(function() {
    window.play_btn.btnf.addEventListener('click', userInterface.playButtonClickListener);
    document.onkeydown = userInterface.onkeydown;
    window.onmousedown = userInterface.onmousedown;
    window.addEventListener('mouseup', userInterface.onmouseup);
    window.onresize = userInterface.onresize;

    // Hide top score
    userInterface.hideTop();

    // Overlays
    userInterface.initOverlays();

    // Load preferences
    userInterface.loadPreference('logDebugging', false);
    userInterface.loadPreference('visualDebugging', false);
    userInterface.loadPreference('autoRespawn', false);
    userInterface.loadPreference('mobileRender', false);
    userInterface.loadPreference('leaderboard', true);
    userInterface.loadPreference('MachineLearning', false);
    window.nick.value = userInterface.loadPreference('savedNick', 'Slither.io-bot');

    // Don't load saved options or apply custom options if
    // the user wants to use default options
    if (typeof(customBotOptions.useDefaults) !== 'undefined'
       && customBotOptions.useDefaults === true) {
        window.log('Ignoring saved / customised options per user request');
    } else {
        // Load saved options, if any
        var savedOptions = userInterface.loadPreference('options', null);
        if (savedOptions !== null) { // If there were saved options
            // Parse the options and overwrite the default bot options
            savedOptions = JSON.parse(savedOptions);
            if (Object.keys(savedOptions).length !== 0
                && savedOptions.constructor === Object) {
                Object.keys(savedOptions).forEach(function(key) {
                    window.bot.opt[key] = savedOptions[key];
                });
            }
            window.log('Found saved settings, overwriting default bot options');
        } else {
            window.log('No saved settings, using default bot options');
        }

        // Has the user customised the options?
        if (Object.keys(customBotOptions).length !== 0
            && customBotOptions.constructor === Object) {
            Object.keys(customBotOptions).forEach(function(key) {
                window.bot.opt[key] = customBotOptions[key];
            });
            window.log('Custom settings found, overwriting current bot options');
        }
    }

    // Save the bot options
    userInterface.savePreference('options', JSON.stringify(window.bot.opt));
    window.log('Saving current bot options');

    // Listener for mouse wheel scroll - used for setZoom function
    document.body.addEventListener('mousewheel', canvasUtil.setZoom);
    document.body.addEventListener('DOMMouseScroll', canvasUtil.setZoom);

    // Set render mode
    if (window.mobileRender) {
        userInterface.toggleMobileRendering(true);
    } else {
        userInterface.toggleMobileRendering(false);
    }
    // Remove laggy logo animation
    userInterface.removeLogo();
    // Unblocks all skins without the need for FB sharing.
    window.localStorage.setItem('edttsg', '1');

    // Remove social
    window.social.remove();

    // Maintain fps
    setInterval(userInterface.framesPerSecond.fpsTimer, 80);

    //initialzie lable map for ML mode.
    bot.restartLabelMap(bot.mapSize);
    bot.direction = {x:0, y:-100};  //this initialization should be (also) in another place

    // Start!
    userInterface.oefTimer();
})();
