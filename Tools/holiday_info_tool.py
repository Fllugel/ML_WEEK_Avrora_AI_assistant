from langchain_core.tools import tool


@tool("holiday_info_tool")
def holiday_info_tool(holiday: str) -> str:
    """
    Цей інструмент повертає інформацію про свята і події та товари, які їм відповідають. Залежно від назви свята (holiday), повертається різна інформація.
    Якщо тебе питають порекомендувати, дати пораду, що придбати, подарити або підходить для свята (holiday), перевіряй наявність товарів з бази даних, що відповідають категоріям з переліку (info), і переліковую тільки ті товари, що є в наявності.
    """

    if holiday == "День народження":
        info = """
        Папір для подарунків
Свічки
Вітальні листівки
Фоторамки
Подарункові пакети
Плюшеві іграшки
Ковпаки для вечірки
Гірлянди
Набори кульок
Декоративні лампочки
        """

    if holiday == "Новий рік":
        info = """
            Ялинкові прикраси
Гірлянди
Снігові кулі
Подарункові коробки
Новорічні чашки
Тематичні серветки
Новорічні шкарпетки
Свічки на батарейках
Святкові сервірувальні прикраси
Новорічні вінки

            """

    if holiday == "День знань":
        info = """
Зошити
Ручки
Рюкзаки
Ланчбокси
Пенали
Лінійки
Стікери
Маркери
Папки
Обкладинки для книг

            """

    if holiday == "Весілля":
        info = """
Фотоальбоми
Фужери для шампанського
Гостьові книги
Весільні сувеніри
Декоративні свічки
Фігурки для торта
Пелюстки квітів
Тримачі для кілець
Конфеті
Тематичні скатертини
            """

    if holiday == "Baby Shower":
        info = """
Одяг для немовлят
М'які іграшки
Ковдри для дітей
Фоторамки для дитячих фото
Подарункові коробки
Іграшки для прорізування зубів
Слинявчики
Дитячі пляшечки
Декор для кімнати
Музичні іграшки

            """

    if holiday == "Випускний":
        info = """
Фотоальбоми
Трофеї
Подарункові сертифікати
Мотиваційні плакати
Декоративні предмети
Вітальні банери
Ручки
Планери
Зошити
Обрамлені сертифікати
            """

    if holiday == "Різдво":
        info = """
Святкові чашки
Подарункові пакети
Тематичні прикраси
Свічки
Шкарпетки для подарунків
Святкові гірлянди
Снігові кулі
Декор для столу
Святкові рушники
Подарункові набори

            """

    if holiday == "Хелловін":
        info = """
Маски
Гарбузові декорації
Гірлянди
Свічки
Тематичні наклейки
Штучна павутина
Костюми
Сумки для солодощів
Ліхтарики
Гарбузові фігурки
            """

    if holiday == "День Святого Валентина":
        info = """
Валентинки
Плюшеві іграшки
Свічки
Шоколадні набори
Рамки для фото
Квіткові вази
Подушки-серця
Ароматичні масла
Кулони або браслети
Подарункові коробки
            """

    if holiday == "Великдень":
        info = """
Декоративні яйця
Кошики
Наклейки для яєць
Тканинні серветки
Фігурки зайчиків
Форми для випічки
Вітальні листівки
Папір для випічки
Святкові рушники
Свічки

            """

    if holiday == "День матері":
        info = """
Квіткові вази
Декоративні свічки
Тематичні листівки
Кухонний текстиль
Шоколадні набори
Ароматичні свічки
Подушки для відпочинку
Подарункові пакети
Набори для чаю або кави
Ароматизатори для дому
            """

    if holiday == "День батька":
        info = """
Кружки з написами
Ремені
Гаманці
Набори інструментів
Фляги
Спортивні пляшки для води
Тематичні футболки
Декоративні рамки
Брелоки
Подарункові коробки
            """

    if holiday == "Ювілей":
        info = """
Пам’ятні фотоальбоми
Гравіровані келихи
Святкові банери
Свічки
Фоторамки
Подарункові коробки
Набори декору для столу
Серветки із тематичними візерунками
Декоративні стрічки
Тематичні листівки
            """

    if holiday == "Річниця весілля":
        info = """
Келихи для шампанського
Декоративні свічки
Рамки для спільних фото
М'які подушки у формі серця
Ковдри
Шоколадні набори
Вази для квітів
Подарункові сертифікати
Набори ароматичних свічок
Листівки з побажаннями
            """
    if holiday == "Перший день у школі":
        info = """
        Зошити з яскравими обкладинками
Ручки
Олівці
Пенали
Лінійки
Ланчбокси
Рюкзаки
Гумки
Клей-олівці
Обкладинки для підручників
        """
    if holiday == "Благодійний захід":
        info = """
Подарункові пакети
Набори канцелярії
Тематичні листівки
Блокноти
Декоративні коробки
Свічки
Подарункові сертифікати
Мотиваційні плакати
Плюшеві іграшки
Тематичний декор

        """
    if holiday == "Пікнік":
        info = """
Одноразовий посуд
Паперові серветки
Пікнікові ковдри
Холодильні пакети
Контейнери для їжі
Сітки для грилю
Пластикові стакани
Набори столових приборів
Термоси
Набори для барбекю
        """
    if holiday == "День святого Миколая":
        info = """
Святкові коробки
Новорічні шкарпетки
Листівки для дітей
Подарункові набори
Іграшки
Тематичні свічки
Шоколадні фігурки
Прикраси для дому
Солодощі
Гірлянди

        """
    if holiday == "День незалежності":
        info = """
Прапори
Декоративні стрічки
Тематичні гірлянди
Браслети в національному стилі
Вишиванки
Тематичні чашки
Святкові кульки
Наліпки
Фоторамки
Банери
        """
    if holiday == "Сімейний збір":
        info = """
Фотоальбоми
Настільні ігри
Святкові скатертини
Набори серветок
Кухонний текстиль
Посуд для сервірування
Келихи для напоїв
Ковдри для пікніка
Декоративні підсвічники
Лампи на батарейках
        """
    if holiday == "Новий автомобіль":
        info = """
Ароматизатори для авто
Чохли для сидінь
Органайзери для багажника
Щітки для очищення скла
Ганчірки для полірування
Підстаканники
Зарядні пристрої для авто
Автомобільні килимки
Наліпки для декору
Декоративні іграшки на дзеркало

        """
    if holiday == "Весняне прибирання":
        info = """
Рукавички для прибирання
Миючі засоби
Спреї для очищення вікон
Серветки для пилу
Органайзери для зберігання
Швабри
Губки
Відра для миття
Повітряні освіжувачі
Щітки для важкодоступних місць
        """
    if holiday == "День бабусі":
        info = """
Вази для квітів
Декоративні свічки
Кухонний текстиль (рушники, прихватки)
Комплекти для рукоділля
Подарункові набори чаю
Зручні подушки для відпочинку
Креми для догляду за шкірою
Тематичні листівки
Рамки для сімейних фото
Зошити для рецептів

        """
    if holiday == "День дідуся":
        info = """
Кружки з написами
Брелоки
Гаманці
Комплекти інструментів
Термоси
Декоративні рамки для фото
Тематичні футболки
Листівки з побажаннями
Набори для догляду за авто
Ароматизатори для дому

        """
    if holiday == "День закоханих":
        info = """
Сердечка-подушки
Ароматичні свічки
Вітальні листівки
Квіткові вази
Тематичні серветки
Шоколадні набори
Комплекти ароматичних масел
Плюшеві іграшки
Декоративні підсвічники
Романтичні постери

        """
    if holiday == "День відкриття магазину":
        info = """
Святкові банери
Гірлянди
Набори повітряних кульок
Подарункові пакети
Декоративні стрічки
Тематичні серветки
Листівки для клієнтів
Ковпаки для вечірок
Паперові стаканчики та тарілки
Святкові наклейки

        """
    if holiday == "Вечір кіно вдома":
        info = """
Пледи для затишку
Миски для попкорну
Набори для приготування попкорну
Декоративні лампочки
Тематичні подушки
Свічки для атмосфери
Набори для приготування напоїв
Кухонний текстиль (серветки, рушники)
Настільні ігри для перерви
Постери для створення атмосфери
        """
    if holiday == "День еколога":
        info = """
Сумки для покупок із тканини
Органайзери для сортування сміття
Еко-чашки
Комплекти для вирощування рослин
Контейнери для багаторазового використання
Багаторазові пляшки для води
Лампи на сонячних батареях
Еко-щітки для прибирання
Серветки з бамбукового волокна
Декоративні вазони

        """
    if holiday == "День переїзду":
        info = """
Коробки для зберігання
Органайзери
Скотч для пакування
Етикетки для маркування
Папір для обгортання
Багаторазові мішки для речей
Рукавички для пакування
Чохли для меблів
Губки та серветки для прибирання
Освіжувачі повітря
        """
    if holiday == "День спорту":
        info = """
Спортивні пляшки для води
Гантелі
Рушники для тренувань
Еластичні стрічки для вправ
Сумки для тренувального одягу
Йога-килими
Фітнес-браслети
Мотиваційні плакати
Комплекти для догляду за тілом після тренувань
Тематичні футболки

        """
    if holiday == "Осінній фестиваль":
        info = """
Гірлянди з осінніми листками
Декоративні гарбузи
Пледи
Свічки з ароматами осені
Кухонний текстиль із осінніми візерунками
Керамічні вази
Набори для випічки
Тематичні серветки
Наклейки з осінньою тематикою
Святкові листівки
        """
    if holiday == "Святковий бранч":
        info = """
Тематичний посуд
Серветки
Скатертини
Чашки з декором
Набори для чаю
Декоративні підставки для столу
Свічки для створення атмосфери
Келихи для напоїв
Прикраси для сервірування
Органайзери для столових приборів
        """
    if holiday == "День музики":
        info = """
Тематичні постери
Набори навушників
Чохли для музичних інструментів
Стильні блокноти для нотаток
Аксесуари для музичних інструментів
Декоративні свічки у формі нот
Брелоки з музичною тематикою
Тематичні футболки
Рамки для фото з музичними візерунками
Подарункові сертифікати

        """
    if holiday == "День друзів":
        info = """
Настільні ігри
Кружки з написами про дружбу
Фотоальбоми для спільних спогадів
Ковдри для затишних вечорів
Листівки з побажаннями
Брелоки у формі сердець або друзів
Подушки із написами
Ароматичні свічки
Набори для чаювання
Тематичні значки

        """
    if holiday == "День народження домашнього улюбленця":
        info = """
Іграшки для тварин
Миски для корму
Лежанки
Одяг для тварин
Щітки для догляду за шерстю
Ласощі
Тематичні бантики
Повідки або нашийники
Ковдри для тварин
Набори для догляду

        """
    if holiday == "День фотографії":
        info = """
Рамки для фото
Альбоми для фотографій
Світлові гірлянди для фото
Магніти для фото
Листівки для підписів
Фотопапір
Декоративні кліпси для кріплення фото
Тематичні стікери
Коробки для зберігання фото
Книжки для скрапбукінгу

        """
    if holiday == "День гумору":
        info = """
Сувенірні футболки з жартами
Маски для розіграшів
Наліпки з гумористичними написами
Тематичні чашки
Жартівливі значки
Настільні ігри з веселими завданнями
Брелоки з мемами
Листівки з жартами
Комічні магніти на холодильник
Іграшки-антистреси

        """
    if holiday == "День кіно":
        info = """
Пледи для перегляду фільмів
Миски для попкорну
Плакати з кінотематикою
Фоторамки у стилі кінострічок
Тематичні подушки
Кружки із символікою улюблених фільмів
Настільні лампи
Набори для приготування попкорну
Брелоки з персонажами
Листівки для кіноманів

        """
    if holiday == "День природи":
        info = """
Сумки для збирання сміття
Еко-пляшки
Комплекти для садівництва
Вази для квітів
Горщики для рослин
Лійки
Рукавички для роботи з землею
Насіння квітів чи овочів
Освітлення на сонячних батареях
Тематичні листівки

        """
    if holiday == "День толерантності":
        info = """
Браслети з символікою рівності
Листівки з мотивуючими написами
Тематичні футболки
Плакати з підтримкою толерантності
Значки з відповідною тематикою
Подарункові коробки
Декоративні свічки
Кружки із надихаючими словами
Альбоми для спільних спогадів
Магніти з мотиваційними висловами

        """
    if holiday == "День здоров'я":
        info = """
Рушники для фітнесу
Йога-килими
Пляшки для води
Набори для догляду за тілом
Масажні ролики
Комплекти вітамінів (сувенірні)
Ароматичні свічки для релаксації
Настільні органайзери для медикаментів
Листівки з порадами для здоров’я
Книжки з рецептами для правильного харчування
        """
    if holiday == "Святкова вечеря":
        info = """
Тематичний посуд
Скатертини
Келихи для напоїв
Декоративні серветки
Підставки для столових приборів
Свічки для атмосфери
Набори для приготування страв
Чашки з дизайном до свята
Подарункові коробки
Органайзери для спецій

        """

    if holiday == "День весни":
        info = """
Вазони для квітів
Набори для садівництва
Ароматичні свічки з весняними запахами
Пледи для пікніка
Декоративні фігурки зайчиків або квітів
Тематичні листівки
Кухонний текстиль із квітковими мотивами
Горщики для посадки рослин
Серветки з весняними візерунками
Світлодіодні лампи у формі квітів

        """

    if holiday == "День кухаря":
        info = """
Фартухи з написами
Набори лопаток та вінчиків
Силіконові форми для випічки
Декоративні баночки для спецій
Рукавички для гарячого
Кухонний таймер
Подарункові серветки
Блокноти для рецептів
Декоративні рушники
Посуд для сервірування
        """

    if holiday == "Відпустка":
        info = """
Валізи або дорожні сумки
Багаторазові пляшки для води
Органайзери для косметики
Сонцезахисні окуляри
Пляжні рушники
Креми для захисту від сонця
Капелюхи
Тематичні футболки
Лампи для кемпінгу
Сувенірні магніти

        """

    if holiday == "День книги":
        info = """
Закладки для книг
Світлодіодні лампи для читання
Тематичні чашки з цитатами
Подарункові набори блокнотів
Настінні постери з улюбленими книгами
Листівки з цитатами авторів
Обкладинки для книг
Тематичні наклейки
Подушки із зображенням книг
Подарункові пакети

        """

    if holiday == "Святковий концерт":
        info = """
Тематичні плакати
Світлодіодні палички
Кружки із символікою улюблених гуртів
Настільні рамки для фото
Значки або браслети
Підставки для квитків
Тематичні листівки
Подушки з музичними принтами
Свічки з ароматом атмосфери
Подарункові коробки

        """

    if holiday == "Перший сніг":
        info = """
Пледи для затишку
Свічки з зимовими ароматами
Тематичні чашки для гарячого шоколаду
Рукавички або шапки
Сніжинки для декору
Наліпки із зимовими мотивами
Ліхтарики на батарейках
Килимки із зимовими принтами
Брелоки у формі сніжинок
Альбоми для зимових фото
        """

    if holiday == "Літній табір":
        info = """
Рюкзаки
Пляшки для води
Пледи для кемпінгу
Ліхтарики для ночівлі
Ланчбокси
Гігієнічні серветки
Захисні креми від сонця
Тематичні футболки
Ковпаки для вечірок біля вогню
Комплекти для творчості
        """

    if holiday == "День молоді":
        info = """
Настільні ігри
Тематичні футболки
Брелоки з мотиваційними словами
Кружки із написами
Постери із трендовими зображеннями
Пледи для пікніка
Гаджети або аксесуари для телефонів
Наліпки для ноутбуків
Пляшки для води
Органайзери для дрібниць
        """

    if holiday == "День незалежності":
        info = """
Національні прапори
Браслети із патріотичною символікою
Настінні плакати
Декоративні серветки
Тематичні футболки
Гірлянди з національними мотивами
Кружки із гербом або прапором
Декоративні магніти
Значки або наліпки
Листівки з патріотичними побажаннями
        """

    if holiday == "Ремонт у домі":
        info = """
Рукавички для роботи
Щітки та губки
Набори фарбувальних пензлів
Клейові пістолети
Органайзери для інструментів
Стрічки для фарбування
Набори гачків та тримачів
Лампочки для освітлення
Текстиль для оновлення кімнат
Килимки для інтер'єру
        """

    if holiday == "День учителя":
        info = """
Тематичні чашки
Листівки з подяками
Набори блокнотів
Подарункові ручки
Тематичні магніти
Декоративні свічки
Серветки із візерунками
Подарункові коробки
Фоторамки для спільних фото
Набори для чаювання
        """
    if holiday == "День тренера":
        info = """
Спортивні рушники
Пляшки для води
Тематичні футболки
Настінні плакати з мотивацією
Органайзери для інструментів
Масажні ролики
Брелоки із символікою спорту
Листівки з подяками
Подарункові сертифікати
Набори для догляду

        """
    if holiday == "День першого побачення":
        info = """
Тематичні свічки
Пледи для романтичної атмосфери
Листівки з побажаннями
Тематичні подушки
Набори для чаювання
Кружки у формі серця
Брелоки для закоханих
Фоторамки для майбутніх спогадів
Ароматичні свічки
Подарункові коробки
        """

    if holiday == "День відкриття бізнесу":
        info = """
Святкові банери
Гірлянди
Повітряні кульки
Набори для сервірування столу
Листівки для клієнтів
Органайзери для канцелярії
Паперові серветки
Декоративні підсвічники
Коробки для сувенірів
Брелоки із символікою бізнесу
        """
    if holiday == "День бабусі й дідуся":
        info = """
Пледи для затишку
Настінні фоторамки
Комплекти для рукоділля
Набори для чаю або кави
Листівки із вдячністю
Тематичні вазони
Декоративні свічки
Подарункові коробки
Текстиль для дому
Брелоки для ключів
        """
    if holiday == "День осіннього листя":
        info = """
Декоративні гарбузи
Серветки із осінніми мотивами
Пледи для затишку
Листівки із тематичними віршами
Світлодіодні лампи у формі листя
Декоративні підсвічники
Горщики для квітів
Набори для садівництва
Тканинні серветки
Настінні прикраси

        """
    if holiday == "День вина":
        info = """
Тематичні келихи
Подарункові набори штопорів
Скатертини для святкових столів
Настінні плакати із винною тематикою
Тематичні підставки для пляшок
Серветки із відповідними мотивами
Подарункові коробки
Набори келихів для дегустації
Листівки з описами вин
Декоративні аксесуари

        """

    if holiday == "День рукоділля":
        info = """
Набори для вишивання
Органайзери для ниток
Тематичні блокноти
Коробки для зберігання матеріалів
Набори інструментів для шиття
Магніти із рукодільними мотивами
Набори для створення браслетів
Подарункові сертифікати
Листівки для рукодільниць
Комплекти для творчості
        """

    if holiday == "День настільних ігор":
        info = """
Набори настільних ігор
Ковдри для затишної атмосфери
Серветки для закусок
Кружки з веселими написами
Листівки із правилами ігор
Брелоки у формі кубиків
Тематичні подушки
Настінні постери з логотипами ігор
Настільні лампи для вечірніх ігор
Коробки для зберігання аксесуарів
        """

    if holiday == "День домашнього затишку":
        info = """
Ароматичні свічки
Пледи
Декоративні подушки
Вазони для квітів
Органайзери для речей
Декоративні лампи
Набори рушників
Брелоки із символікою дому
Листівки для побажань
Тематичні магніти
        """

    if holiday == "День книги":
        info = """
Закладки для книг
Світлодіодні лампи для читання
Тематичні чашки із цитатами
Блокноти для нотатків
Листівки з улюбленими цитатами
Обкладинки для книг
Декоративні магніти із зображенням книг
Органайзери для зберігання книг
Тематичні постери
Подарункові коробки

        """

    if holiday == "День фотографа":
        info = """
Рамки для фото
Фотокліпси
Альбоми для фотографій
Настінні прикраси для фото
Листівки із тематикою фотографії
Декоративні коробки для зберігання фото
Набори для скрапбукінгу
Брелоки у формі камер
Світлодіодні гірлянди для фото
Тематичні чашки

        """

    if holiday == "День Землі":
        info = """
Сумки для багаторазового використання
Еко-пляшки для води
Горщики для рослин
Органайзери для сортування сміття
Набори для вирощування квітів
Лампи на сонячних батареях
Листівки з порадами для екологічного життя
Ароматичні свічки із натуральних матеріалів
Наліпки з екологічною тематикою
Тематичні постери
        """
    if holiday == "День вишиванки":
        info = """
⦁	Вишиті серветки
⦁	Настільні рушники
⦁	Тематичні браслети
⦁	Листівки з візерунками
⦁	Сумки з вишивкою
⦁	Брелоки із традиційними орнаментами
⦁	Подарункові коробки
⦁	Настінні прикраси із національними мотивами
⦁	Плакати із традиційними символами
⦁	Декоративні подушки із вишивкою
        """
    if holiday == "День кохання":
        info = """
⦁	Сердечка-подушки
⦁	Набори ароматичних свічок
⦁	Тематичні листівки
⦁	Декоративні рамки для фото
⦁	Шоколадні подарункові набори
⦁	Вазони для квітів
⦁	Набори для створення листівок
⦁	Чашки у формі серця
⦁	Брелоки для пар
⦁	Подарункові коробки
        """
    if holiday == "День молоді":
        info = """
⦁	Тематичні футболки
⦁	Брелоки із веселими написами
⦁	Набори настільних ігор
⦁	Органайзери для гаджетів
⦁	Сумки для шопінгу із принтами
⦁	Пляшки для води
⦁	Тематичні постери
⦁	Кружки із мотиваційними цитатами
⦁	Наліпки для ноутбуків
⦁	Подарункові сертифікати
        """
    if holiday == "Осінній пікнік":
        info = """
⦁	Термоси
⦁	Пледи
⦁	Набори посуду для пікніка
⦁	Серветки із осінніми мотивами
⦁	Декоративні гарбузи
⦁	Коробки для закусок
⦁	Ліхтарики на батарейках
⦁	Тематичні гірлянди
⦁	Чашки із принтами осені
⦁	Набори для барбекю
        """
    if holiday == "День зимових свят":
        info = """
⦁	Святкові гірлянди
⦁	Декоративні свічки
⦁	Чашки із зимовими принтами
⦁	Тематичні листівки
⦁	Настінні прикраси у формі сніжинок
⦁	Коробки для подарунків
⦁	Декоративні підставки для столу
⦁	Пледи для затишку
⦁	Брелоки у формі зимових символів
⦁	Магніти із зимовими мотивами
        """
    if holiday == "День подяки":
        info = """
⦁	Тематичні серветки
⦁	Скатертини із святковими мотивами
⦁	Чашки із написами вдячності
⦁	Набори для сервірування столу
⦁	Листівки із побажаннями
⦁	Подарункові коробки
⦁	Тематичні гірлянди
⦁	Декоративні свічки
⦁	Плакати з написами "Дякую"
⦁	Настільні прикраси
        """

    if holiday == "Вечір спогадів":
        info = """
⦁	Фотоальбоми
⦁	Рамки для фото
⦁	Листівки із теплими словами
⦁	Пледи для затишку
⦁	Чашки для гарячого шоколаду
⦁	Декоративні свічки
⦁	Набори для скрапбукінгу
⦁	Книжки для запису спогадів
⦁	Органайзери для зберігання фото
⦁	Магніти із тематикою пам’яті
        """
    if holiday == "День спорту":
        info = """
⦁	Спортивні пляшки для води
⦁	Рушники для тренувань
⦁	Еластичні стрічки для вправ
⦁	Йога-килими
⦁	Тематичні футболки
⦁	Настінні мотиваційні плакати
⦁	Масажні ролики
⦁	Брелоки із символікою спорту
⦁	Листівки з побажаннями успіхів
⦁	Сумки для спортивного одягу

        """
    if holiday == "Літній фестиваль":
        info = """
⦁	Сонцезахисні окуляри
⦁	Сумки для пляжу
⦁	Гірлянди для декору
⦁	Тематичні футболки
⦁	Пляшки для води
⦁	Капелюхи
⦁	Серветки із літніми мотивами
⦁	Лампи на батарейках
⦁	Наліпки із символікою фестивалю
⦁	Брелоки
        """
    if holiday == "Святковий ярмарок":
        info = """
⦁	Набори для випічки
⦁	Тематичні листівки
⦁	Скатертини для сервірування
⦁	Кухонний текстиль
⦁	Гірлянди із сезонними мотивами
⦁	Кружки із тематичними принтами
⦁	Коробки для подарунків
⦁	Світлодіодні прикраси
⦁	Декоративні свічки
⦁	Пледи
        """
    if holiday == "Річниця компанії":
        info = """
⦁	Тематичні банери
⦁	Корпоративні листівки
⦁	Подарункові ручки
⦁	Альбоми для фото співробітників
⦁	Тематичні чашки із логотипом
⦁	Брелоки для співробітників
⦁	Подарункові пакети
⦁	Кухонний текстиль із корпоративними кольорами
⦁	Настільні органайзери
⦁	Сертифікати подяки
        """
    if holiday == "День дітей":
        info = """
⦁	М’які іграшки
⦁	Настільні ігри
⦁	Розмальовки
⦁	Набори олівців та фломастерів
⦁	Листівки з казковими персонажами
⦁	Подарункові пакети
⦁	Тематичні подушки
⦁	Набори для творчості
⦁	Коробки для іграшок
⦁	Брелоки із дитячими персонажами
        """
    if holiday == "Романтичний вечір":
        info = """
⦁	Свічки із романтичними ароматами
⦁	Пледи для затишку
⦁	Рамки для фото
⦁	Тематичні подушки
⦁	Листівки із теплими побажаннями
⦁	Набори для чаювання
⦁	Кухонний текстиль із вишуканими мотивами
⦁	Шоколадні подарункові набори
⦁	Брелоки для пар
⦁	Вазони для квітів
        """
    if holiday == "День подорожей":
        info = """
⦁	Рюкзаки
⦁	Багаторазові пляшки для води
⦁	Органайзери для документів
⦁	Капелюхи для подорожей
⦁	Сонцезахисні окуляри
⦁	Брелоки із символами подорожей
⦁	Листівки із мотивуючими написами
⦁	Тематичні чашки
⦁	Пледи для довгих поїздок
⦁	Контейнери для їжі
        """
    if holiday == "Зимовий пікнік":
        info = """
⦁	Термоси
⦁	Пледи для тепла
⦁	Набори одноразового посуду
⦁	Свічки для створення атмосфери
⦁	Серветки із зимовими мотивами
⦁	Гірлянди на батарейках
⦁	Чашки для гарячих напоїв
⦁	Коробки для закусок
⦁	Рукавички для пікніка
⦁	Набори для приготування їжі
        """
    if holiday == "День домашнього улюбленця":
        info = """
⦁	Іграшки для тварин
⦁	Лежанки
⦁	Миски для їжі
⦁	Одяг для домашніх улюбленців
⦁	Повідки
⦁	Щітки для догляду за шерстю
⦁	Набори ласощів
⦁	Коробки для зберігання корму
⦁	Брелоки із символікою тварин
⦁	Тематичні подушки
        """
    if holiday == "Вечір креативності":
        info = """
⦁	Набори для малювання
⦁	Альбоми для творчості
⦁	Розмальовки
⦁	Набори для створення прикрас
⦁	Стійки для картин
⦁	Пензлі та фарби
⦁	Набори для скрапбукінгу
⦁	Декоративні рамки
⦁	Тематичні листівки
⦁	Коробки для зберігання матеріалів
        """
    if holiday == "День першого снігу":
        info = """

        """
    if holiday == "":
        info = """
⦁	Свічки із зимовими ароматами
⦁	Пледи для затишку
⦁	Гірлянди у формі сніжинок
⦁	Тематичні чашки для гарячого шоколаду
⦁	Брелоки у вигляді сніговиків
⦁	Настінні прикраси із зимовими мотивами
⦁	Коробки для подарунків
⦁	Серветки з зимовими візерунками
⦁	Листівки із теплими побажаннями
⦁	Декоративні сніжинки
        """
    if holiday == "День бабусі":
        info = """
⦁	Вази для квітів
⦁	Набори для рукоділля
⦁	Подушки для відпочинку
⦁	Кухонний текстиль (рушники, прихватки)
⦁	Листівки з подяками
⦁	Чайні набори
⦁	Декоративні свічки
⦁	Рамки для фото
⦁	Книжки для запису рецептів
⦁	Тематичні браслети
        """
    if holiday == "День дідуся":
        info = """
⦁	Настільні органайзери
⦁	Тематичні футболки
⦁	Брелоки для ключів
⦁	Кружки з написами про дідуся
⦁	Листівки з теплими словами
⦁	Комплекти інструментів
⦁	Ковдри для затишку
⦁	Настінні постери
⦁	Подарункові коробки
⦁	Тематичні магніти
        """
    if holiday == "Вечір читання":
        info = """
⦁	Пледи для затишку
⦁	Лампи для читання
⦁	Закладки для книг
⦁	Настільні органайзери для книг
⦁	Чашки із цитатами
⦁	Листівки із книжковими мотивами
⦁	Брелоки у формі книг
⦁	Декоративні рамки для фото
⦁	Книжкові підставки
⦁	Тематичні наклейки
        """
    if holiday == "День толерантності":
        info = """
⦁	Браслети із символікою рівності
⦁	Тематичні футболки
⦁	Листівки з мотивуючими словами
⦁	Плакати із закликами до толерантності
⦁	Наліпки із символікою миру
⦁	Тематичні чашки
⦁	Магніти з позитивними написами
⦁	Свічки із надихаючими словами
⦁	Брелоки із символікою рівності
⦁	Настінні постери
        """
    if holiday == "День природи":
        info = """
⦁	Вазони для кімнатних рослин
⦁	Горщики для квітів
⦁	Комплекти для садівництва
⦁	Сумки для сортування сміття
⦁	Набори для вирощування рослин
⦁	Тематичні листівки
⦁	Освітлення на сонячних батареях
⦁	Серветки із природними мотивами
⦁	Органайзери для зберігання речей
⦁	Еко-пляшки
        """
    if holiday == "Весняне прибирання":
        info = """
⦁	Рукавички для прибирання
⦁	Миючі засоби
⦁	Серветки для пилу
⦁	Швабри та відра
⦁	Освіжувачі повітря
⦁	Органайзери для речей
⦁	Паперові рушники
⦁	Щітки для важкодоступних місць
⦁	Кухонні губки
⦁	Пакети для сміття
        """
    if holiday == "Літній кемпінг":
        info = """
⦁	Ліхтарики на батарейках
⦁	Пледи для кемпінгу
⦁	Сумки для зберігання речей
⦁	Термоси
⦁	Набори для барбекю
⦁	Пляшки для води
⦁	Серветки для пікніка
⦁	Контейнери для їжі
⦁	Ковпаки для вечірок біля вогню
⦁	Гамачки
        """
    if holiday == "День вдячності":
        info = """
⦁	Листівки з написами "Дякую"
⦁	Тематичні чашки
⦁	Подарункові коробки
⦁	Серветки із вдячними мотивами
⦁	Настінні прикраси із теплими словами
⦁	Браслети із надихаючими написами
⦁	Плакати для святкового декору
⦁	Набори чаю чи кави
⦁	Декоративні свічки
⦁	Рамки для фото
        """
    if holiday == "Вечір настільних ігор":
        info = """
⦁	Набори настільних ігор
⦁	Тематичні килимки для столу
⦁	Чашки із веселими написами
⦁	Серветки для закусок
⦁	Брелоки у формі ігрових фігур
⦁	Настінні постери з логотипами ігор
⦁	Органайзери для зберігання ігор
⦁	Пледи для затишної атмосфери
⦁	Тематичні листівки
⦁	Настільні лампи для вечірніх ігор
        """
    if holiday == "День волонтера":
        info = """
⦁	Листівки з подяками
⦁	Тематичні футболки
⦁	Настінні плакати з мотивацією
⦁	Подарункові сертифікати
⦁	Брелоки з написами "Допомагати – це круто"
⦁	Органайзери для канцелярії
⦁	Еко-сумки для покупок
⦁	Тематичні значки
⦁	Декоративні свічки
⦁	Магніти із символами допомоги
        """
    if holiday == "День рукоділля":
        info = """
⦁	Набори для вишивання
⦁	Органайзери для зберігання ниток
⦁	Набори для створення прикрас
⦁	Тематичні блокноти для ідей
⦁	Коробки для зберігання матеріалів
⦁	Листівки із рукодільними мотивами
⦁	Декоративні рамки для готових робіт
⦁	Пензлі та фарби для творчості
⦁	Набори для скрапбукінгу
⦁	Тематичні магніти
        """
    if holiday == "День спогадів":
        info = """
⦁	Фотоальбоми
⦁	Рамки для фото
⦁	Настільні органайзери для зберігання пам’яток
⦁	Листівки із теплими побажаннями
⦁	Тематичні чашки для чаювання
⦁	Свічки для затишної атмосфери
⦁	Декоративні коробки для пам’ятних речей
⦁	Пледи для вечора спогадів
⦁	Плакати з мотиваційними цитатами
⦁	Брелоки у формі сердець
        """
    if holiday == "День гарбуза":
        info = """
⦁	Декоративні гарбузи
⦁	Свічки у формі гарбуза
⦁	Серветки із осінніми мотивами
⦁	Настінні прикраси із осінніми візерунками
⦁	Набори для вирізання гарбузів
⦁	Листівки із тематикою осені
⦁	Тематичні гірлянди
⦁	Набори для сервірування столу
⦁	Кухонний текстиль із гарбузовими принтами
⦁	Коробки для подарунків
        """
    if holiday == "День миру":
        info = """
⦁	Плакати з символами миру
⦁	Листівки із побажаннями гармонії
⦁	Тематичні браслети
⦁	Свічки із надихаючими написами
⦁	Значки з символікою миру
⦁	Чашки із написами "Мир у всьому світі"
⦁	Набори для малювання із мирними мотивами
⦁	Магніти із символами голуба миру
⦁	Брелоки із символікою рівності
⦁	Тематичні постери
        """
    if holiday == "День кіно":
        info = """
⦁	Настінні плакати із постерами улюблених фільмів
⦁	Пледи для затишного вечора кіно
⦁	Миски для попкорну
⦁	Кружки із зображенням героїв фільмів
⦁	Декоративні лампи у формі кінокамери
⦁	Серветки із кінотематикою
⦁	Листівки із цитатами з фільмів
⦁	Брелоки у формі кінохлопавки
⦁	Настільні органайзери для зберігання дисків
⦁	Тематичні подушки
        """
    if holiday == "День здоров'я":
        info = """
⦁	Йога-килими
⦁	Пляшки для води
⦁	Масажні ролики
⦁	Набори для догляду за тілом
⦁	Рушники для тренувань
⦁	Настінні мотиваційні плакати
⦁	Органайзери для медикаментів
⦁	Листівки із порадами для здорового способу життя
⦁	Подарункові сертифікати
⦁	Ароматичні свічки
        """
    if holiday == "День нових починань":
        info = """
⦁	Блокноти для планування
⦁	Настільні органайзери
⦁	Плакати із мотиваційними написами
⦁	Тематичні ручки
⦁	Листівки із побажаннями успіху
⦁	Кружки з надихаючими цитатами
⦁	Подарункові коробки
⦁	Свічки для затишної атмосфери
⦁	Магніти із позитивними написами
⦁	Брелоки з написом "Новий старт"
        """

    return info
