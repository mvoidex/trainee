{-# LANGUAGE RankNTypes, TemplateHaskell, MultiParamTypeClasses, FlexibleInstances, UndecidableInstances, DataKinds, KindSignatures, ScopedTypeVariables #-}

module Numeric.Trainee.Types (
	Gradee(..),
	Ow(..),
	Named(..), nameIt, unnameIt, name,
	Learnee(..), params, learneePass, Computee,
	Cost,
	Params(..), sumP
	) where

import Prelude hiding (id, (.))
import Prelude.Unicode

import qualified Control.Arrow as A (second)
import Control.Category
import Control.Lens
import Data.Proxy
import Data.Tagged
import GHC.TypeLits (KnownSymbol, symbolVal, Symbol)

data Gradee a b = Gradee {
	runGradee ∷ Lens' a b }

instance Category Gradee where
	id = Gradee id
	Gradee f . Gradee g = Gradee (g . f)

class Category a ⇒ Ow a where
	first ∷ a b c → a (b, d) (c, d)
	first = flip stars id

	second ∷ a b c → a (d, b) (d, c)
	second = stars id

	stars ∷ a b c → a b' c' → a (b, b') (c, c')

instance Ow Gradee where
	stars (Gradee f) (Gradee g) = Gradee $ lens g' s' where
		g' (x, y) = (view f x, view g y)
		s' (x, y) (x', y') = (set f x' x, set g y' y)

newtype Named (s ∷ Symbol) w = Named (Tagged s w)

nameIt ∷ Proxy s → w → Named s w
nameIt _ = Named ∘ Tagged

unnameIt ∷ Named s w → w
unnameIt (Named x) = untag x

instance (KnownSymbol s, Show w) ⇒ Show (Named s w) where
	show (Named (Tagged v)) = unwords [symbolVal (Proxy ∷ Proxy s), show v]

data Learnee w a b = Learnee {
	_params ∷ w,
	_learneePass ∷ w → a → (b, b → (a, w)) }

makeLenses ''Learnee

name ∷ Proxy s → Learnee w a b → Learnee (Named s w) a b
name p (Learnee ws f) = Learnee (nameIt p ws) f' where
	f' ws' x = (x', A.second (nameIt p) ∘ back) where
		(x', back) = f (unnameIt ws') x

type Computee a b = Learnee () a b

type Cost b = b → b → (b, b)

class Params w where
	plusP ∷ w → w → w

instance {-# OVERLAPS #-} Params () where
	plusP _ _ = ()

instance Num w ⇒ Params w where
	plusP = (+)

instance {-# OVERLAPS #-} (Params w, Params w') ⇒ Params (w, w') where
	plusP (l, l') (r, r') = (plusP l r, plusP l' r')

instance Params w ⇒ Params (Named s w) where
	plusP l r = nameIt (Proxy ∷ Proxy s) $ unnameIt l `plusP` unnameIt r

sumP ∷ Params w ⇒ [w] → w
sumP = foldr1 plusP
